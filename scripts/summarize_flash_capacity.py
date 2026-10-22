#!/usr/bin/env python3
"""Reproducible planning arithmetic from actual capacity receipts, not a train config."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
s=json.loads(a.suite.read_text());assert s['status']=='completed'
rows=[]
for d in s['reports']:
 assert d['status']=='capacity_preflight_pass'
 b=d['batch'];fw=d['median_rollout_seconds']/b;decode=d['median_decode_seconds']/b
 back=[r['seconds'] for r in d['phases'] if r['phase'].startswith('backward-')]
 backward=statistics.median(back[1:] or back)/b
 rows.append({'batch':b,'rollout_seconds_per_video':fw,'decode_seconds_per_video':decode,'backward_probe_seconds_per_video':backward,'rollout_decode_videos_per_minute':d['rollout_decode_videos_per_minute'],'max_allocated_gib':max(r['peak_allocated_gib'] for r in d['phases']),'max_reserved_gib':max(r['peak_reserved_gib'] for r in d['phases']),'observed_card_peak_used_gib':d['observed_card_peak_used_gib'],'observed_min_free_gib':d['observed_min_free_gib'],'host_peak_rss_gib':d['host_peak_rss_gib']})
base=next(r for r in rows if r['batch']==1)
for r in rows:r['throughput_gain_percent_vs_batch1']=(r['rollout_decode_videos_per_minute']/base['rollout_decode_videos_per_minute']-1)*100
r=next(r for r in rows if r['batch']==s['largest_tested_safe_batch'])
workloads=[]
for commits,train_videos,eval_videos,label in [(596,114432,4500,'public code upper ceiling, NOT confirmed paper endpoint'),(140,26880,1200,'engineering reference near README curve scale, NOT a confirmed exact paper endpoint')]:
 train=train_videos*(r['rollout_seconds_per_video']+r['decode_seconds_per_video']+r['backward_probe_seconds_per_video'])/86400
 evaluation=eval_videos*(r['rollout_seconds_per_video']*50/20+r['decode_seconds_per_video'])/86400
 frames=(train_videos+eval_videos)*81
 workloads.append({'optimizer_commits':commits,'training_videos':train_videos,'evaluation_videos':eval_videos,'interpretation':label,'paper_exact_workload_confirmed':False,'model_train_compute_proxy_days':train,'model_eval_compute_proxy_days':evaluation,'model_compute_proxy_total_days':train+evaluation,'not_a_measured_total_training_time':True,'logical_hps_frame_scores':frames,'hypothetical_reward_extra_days':{str(t):frames*t/86400 for t in [.05,.1,.2]},'reward_seconds_per_frame_were_measured':False})
result={'status':'capacity_tests_complete_plan_pending_user_review','capacity_scope':'real VisualRL model ports; no full trainer/G3 or real reward integration exercised','formal_training_started':False,'rows':rows,'largest_tested_safe_group_divisor_batch':s['largest_tested_safe_batch'],'within_3percent_best_throughput_smallest_batch':s['throughput_preferred_batch'],'skipped':s['skipped'],'proposed_rollout_microbatch_ceiling':4,'proposed_recompute_geometry':'match the recorded rollout forward partitions; do not shrink without actual runtime parity gate','proposed_decode_microbatch':1,'formal_configuration_ready':False,'workload_options_not_automatically_selected':workloads,'extrapolation_caveats':['Backward uses a memory-probe scalar loss, not a measured full PPO update.','Eval50 cost is extrapolated from20 forward steps, not a measured full300-prompt evaluation.','Real reward, JPEG/RPC, filesystem, preprocessing, framework scheduling, EMA/checkpoint and shared-GPU pauses add costs.','Hypothetical reward timings are sensitivity inputs, not benchmark results.','Exact paper endpoint is not recoverable from GPU-hours or plot pixels.']}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
