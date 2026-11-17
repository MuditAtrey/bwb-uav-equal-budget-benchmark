Pool-size experiment, corrected layout (2026-11-16).
run_B12.sh had a bug: its helper run() reused the shell variable n (line count), so only the first seed of each lane received the intended POOL_N;
every later seed ran with POOL_N=288. Evidence: the runs for those seeds are byte-identical across pool_size/*_300, *_1000, *_3000, and a fresh
POOL_N=288 run reproduces the first batch of pool_size/turbo_300/turbo_seed1.csv exactly (POOL_N=300 does not).
pool_size/ is left untouched as run. Here the runs are filed under the pool size they really used:
  *_288   copied from pool_size (turbo seeds 1-4,6-9; adapt10 seeds 1-9; AL-R seeds 1-4); remaining seeds run by run_B12b.sh
  *_1000, *_3000   first-of-lane seeds copied from pool_size (turbo 0,5; adapt10 0; AL-R 0); remaining seeds run by run_B12b.sh
  unused_300   the three runs that really used POOL_N=300; not used in the paper
