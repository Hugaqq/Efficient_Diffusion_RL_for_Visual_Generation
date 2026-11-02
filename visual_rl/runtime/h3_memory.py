"""Opt-in bounded allocator event journal with lossless peak-reset bookkeeping.
No empty_cache, allocator tuning, precision changes or distributed operations.
Spans are hierarchical; every reset follows a persisted peak snapshot. All
CUDA statistics are process-local; NVML/card/process samples are separate.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import datetime,json,os,threading,time


class TorchCounters:
    def __init__(self,device):self.device=device
    def snapshot(self):
        import torch
        if not torch.cuda.is_initialized():return None
        p=torch.cuda.get_device_properties(self.device)
        # PyTorch 2.8's _CUuuid renders without NVML's GPU- prefix. Use the
        # same strict normalization as runtime admission, not string equality
        # against the CUDA_VISIBLE_DEVICES spelling.
        from visual_rl.runtime.h3_session import _uuid
        value=getattr(p,'uuid',None)
        cuda_uuid_type=getattr(torch._C,'_CUuuid',())
        observed_uuid=_uuid(str(value) if isinstance(value,cuda_uuid_type) else value)
        return dict(allocated=torch.cuda.memory_allocated(self.device),reserved=torch.cuda.memory_reserved(self.device),
                    max_allocated=torch.cuda.max_memory_allocated(self.device),max_reserved=torch.cuda.max_memory_reserved(self.device),
                    device_uuid=observed_uuid)
    def reset(self):
        import torch
        torch.cuda.reset_peak_memory_stats(self.device)
    def synchronize(self):
        import torch
        torch.cuda.synchronize(self.device)


class Journal:
    def __init__(self,directory,*,rank,role,device,counters=None,max_bytes=64*1024**2):
        self.rank=rank;self.role=role;self.device=device;self.pid=os.getpid()
        self.counters=counters or TorchCounters(device);self.limit=max_bytes
        self.lock=threading.RLock();self.stack=[];self.base={};self.seq=0;self.reset_id=0
        self.hwm=dict(allocated=0,reserved=0);self.healthy=True;self.bytes=0
        root=Path(directory);root.mkdir(parents=True,exist_ok=True)
        self.path=root/f'{role}-rank{rank}-pid{self.pid}.jsonl';self.stream=self.path.open('x',buffering=1)
        visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
        self.expected_uuid=visible[device] if 0<=device<len(visible) else None
        self.event('process_start',flush=True)
    def _write(self,event,*,flush=True):
        data=(json.dumps(event,allow_nan=False,separators=(',',':'))+'\n').encode()
        if self.bytes+len(data)>self.limit:
            self.healthy=False;raise OSError('memory event log byte budget exhausted')
        try:
            self.stream.write(data.decode());self.stream.flush()
            if flush:os.fsync(self.stream.fileno())
            self.bytes+=len(data)
        except BaseException:
            self.healthy=False;raise
    def _snapshot(self,synchronize=False):
        synced=False;sync_seconds=0.
        if synchronize and threading.current_thread() is threading.main_thread():
            stats=self.counters.snapshot()
            if stats is not None:
                started=time.monotonic();self.counters.synchronize();sync_seconds=time.monotonic()-started;synced=True
        stats=self.counters.snapshot()
        if stats is not None:
            self.hwm['allocated']=max(self.hwm['allocated'],stats['max_allocated'])
            self.hwm['reserved']=max(self.hwm['reserved'],stats['max_reserved'])
        return stats,synced,sync_seconds
    def event(self,kind,*,synchronize=False,flush=True,**fields):
        with self.lock:
            stats,synced,seconds=self._snapshot(synchronize)
            cpu_rss=None
            try:
                for line in Path('/proc/self/status').read_text().splitlines():
                    if line.startswith('VmRSS:'):cpu_rss=int(line.split()[1])*1024;break
            except OSError:pass
            context=dict(self.base)
            for _,frame in self.stack:context.update(frame)
            event=dict(schema='h3.memory-event.v1',seq=self.seq,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                monotonic_ns=time.monotonic_ns(),pid=self.pid,rank=self.rank,role=self.role,device=self.device,
                expected_uuid=self.expected_uuid,thread=threading.current_thread().name,kind=kind,
                phase=self.stack[-1][0] if self.stack else 'outside',ancestors=[s[0] for s in self.stack],
                context=context,counters=stats,whole_run_high_water=dict(self.hwm),reset_id=self.reset_id,
                synchronized=synced,synchronization_seconds=seconds,cpu_rss_bytes=cpu_rss,**fields)
            self._write(event,flush=flush);self.seq+=1
            return event
    def _reset(self,reason,*,synchronize=False):
        # This event MUST reach durable storage before the CUDA counters reset.
        event=self.event('before_unified_reset',synchronize=synchronize,reason=reason,flush=True)
        if event['counters'] is not None:
            self.counters.reset();self.reset_id+=1
            self.event('after_unified_reset',reason=reason,flush=True)
    @contextmanager
    def span(self,phase,*,synchronize=False,**context):
        with self.lock:
            self.event('segment_end',synchronize=synchronize,next_phase=phase)
            self._reset('enter:'+phase)
            self.stack.append((phase,context));self.event('segment_begin',flush=True)
        started=time.monotonic()
        try:yield self
        except BaseException as exc:
            try:self.event('phase_exception',exception_type=type(exc).__name__,message=str(exc)[:8192])
            except BaseException:pass
            raise
        finally:
            # Preserve the original application/OOM error if emergency journal
            # IO or CUDA statistics fail too. Never reset after a failed write.
            import sys
            primary=sys.exc_info()[1]
            try:
                with self.lock:
                    self.event('segment_end',synchronize=synchronize,wall_seconds=time.monotonic()-started)
                    self._reset('leave:'+phase)
                    if not self.stack or self.stack[-1][0]!=phase:
                        self.healthy=False;raise RuntimeError('memory phase stack mismatch')
                    self.stack.pop();self.event('segment_resume')
            except BaseException as cleanup:
                self.healthy=False
                if primary is None:raise
                if hasattr(primary,'add_note'):primary.add_note('memory cleanup failed: '+repr(cleanup))
    def exception(self,exc):
        import torch
        dump=dict(type=type(exc).__name__,message=str(exc)[:16384],pid=self.pid,rank=self.rank)
        if torch.cuda.is_initialized():
            try:dump['allocator_stats']=torch.cuda.memory_stats(self.device)
            except Exception as error:dump['allocator_stats_error']=repr(error)
            try:dump['allocator_summary']=torch.cuda.memory_summary(self.device)[:131072]
            except Exception as error:dump['allocator_summary_error']=repr(error)
        p=self.path.with_suffix('.exception.json');p.write_text(json.dumps(dump,indent=2))
        self.event('exception_exit',exception_type=type(exc).__name__,message=str(exc)[:8192],dump=str(p))
    def assert_healthy(self):
        if not self.healthy or self.stream.closed or not self.path.is_file() or self.bytes<1:
            raise RuntimeError('memory journal unhealthy; do not continue the next update')
        stats=self.counters.snapshot()
        if stats is None or stats['device_uuid']=='unknown':
            raise RuntimeError('memory journal CUDA counters/device UUID unavailable')
        if self.expected_uuid and self.expected_uuid.startswith('GPU-') and str(stats['device_uuid'])!=self.expected_uuid:
            raise RuntimeError('memory journal CUDA UUID differs from explicit rank mapping')
    def close(self):
        with self.lock:
            if self.stream.closed:return
            self.event('process_end',flush=True);self.stream.close()


_instance=None

def enabled():return bool(os.environ.get('H3_MEMORY_DIR'))

def journal():
    global _instance
    if _instance is None and enabled():
        rank=int(os.environ.get('RANK','-1'));role=os.environ.get('H3_MEMORY_ROLE','train')
        device=int(os.environ.get('LOCAL_RANK','0'))
        _instance=Journal(os.environ['H3_MEMORY_DIR'],rank=rank,role=role,device=device)
    return _instance


def event(kind,**fields):
    value=journal()
    if value is not None:return value.event(kind,**fields)


@contextmanager
def span(phase,**fields):
    value=journal()
    if value is None:yield None
    else:
        with value.span(phase,**fields):yield value


def set_context(**fields):
    value=journal()
    if value is not None:
        with value.lock:value.base.update(fields)


def close():
    value=journal()
    if value is not None:value.close()
