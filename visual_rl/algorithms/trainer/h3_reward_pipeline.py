"""Owner-local CPU consumption; two total payload permits, one queued packet.

The caller's existing auxiliary worker remains the sole CUDA producer. The
consumer never enters model/PG work. In-flight HTTP/export cannot be killed:
join waits for the original operation's timeout; the outer attempt deadline
remains authoritative. No partial rows or request retries on failure.
"""
from dataclasses import dataclass
import math
from queue import Empty, Full, Queue
from threading import Event, Lock, Semaphore, Thread

from visual_rl.runtime import h3_memory as memory


@dataclass(frozen=True)
class RewardSample:
    ordinal: int
    sample_id: str
    group_id: str
    member: int
    prompt: str
    selected_index: int


def reward_pipeline(samples, *, decode, score, media_bytes, decode_bytes):
    """Consume FIFO original media storage, completely drain/join before return."""
    if any(type(n) is not int or n <= 0 for n in (media_bytes, decode_bytes)):
        raise ValueError('explicit positive pipeline byte budgets required')
    from visual_rl.models.implementations.h3_loading import _admit_host_memory
    _admit_host_memory(decode_bytes + media_bytes)
    cpu_scope = memory.capture_cpu_scope()
    queue = Queue(maxsize=1)
    permits = Semaphore(2)  # acquired BEFORE decode, released AFTER close
    cancelled, finished = Event(), Event()
    errors, rows = [], []
    error_lock = Lock()

    def fail(error):
        with error_lock:
            if not errors:
                errors.append(error)
            elif error is not errors[0]:
                errors[0].add_note('pipeline secondary failure: ' + repr(error))
            cancelled.set()

    def dispose(media):
        try:
            media.close()
        except BaseException as error:
            fail(error)
        finally:
            permits.release()

    def consume():
        packet = None
        try:
            with cpu_scope():
                while not cancelled.is_set():
                    try:
                        packet = queue.get(timeout=.05)
                    except Empty:
                        if finished.is_set():
                            break
                        continue
                    sample, media = packet
                    try:
                        if not cancelled.is_set():
                            value = score(sample, media)
                            if type(value) is not float or not math.isfinite(value):
                                raise ValueError('reward must be a finite Python float')
                            rows.append((sample.sample_id, sample.group_id, sample.member, value, True))
                    except BaseException as error:
                        fail(error)
                    finally:
                        dispose(media)
                        packet = media = sample = None
        except BaseException as error:
            fail(error)

    consumer = Thread(target=consume, name='h3-reward-consumer', daemon=True)
    consumer.start()
    try:
        for ordinal, sample in enumerate(samples):
            if sample.ordinal != ordinal:
                raise ValueError('pipeline sample ordinal mismatch')
            while not cancelled.is_set():
                if permits.acquire(timeout=.05):
                    break
            else:
                break
            owned = None
            queued = False
            try:
                if cancelled.is_set():
                    break
                with decode(sample) as media:
                    owned = media.handoff()
                payload_bytes = sum(t.numel() * t.element_size() for t in (owned.video, owned.audio))
                if payload_bytes > media_bytes:
                    raise ValueError('pipeline media byte budget exceeded')
                while not cancelled.is_set():
                    try:
                        queue.put((sample, owned), timeout=.05)
                        queued = True
                        break
                    except Full:
                        continue
            except BaseException as error:
                # Record the business failure BEFORE close can fail too.
                fail(error)
                raise
            finally:
                if not queued:
                    if owned is not None:
                        dispose(owned)
                    else:
                        permits.release()
                owned = media = None
    except BaseException as error:
        fail(error)
    finally:
        finished.set()
        # Never close the shared client or a media owner still being consumed.
        consumer.join()
        while True:
            try:
                _, media = queue.get_nowait()
            except Empty:
                break
            dispose(media)
            media = None
    if errors:
        raise errors[0]
    if len(rows) != len(samples):
        raise RuntimeError('pipeline did not consume the complete sample table')
    return tuple(rows)
