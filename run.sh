#!/bin/bash

# 실험 클래스 (ImageNet 어려운 클래스 10개)
CLASSES=(130 289 397 985 292 339 814 417 207 974)

# 출력 폴더
OUTDIR="samples"
mkdir -p $OUTDIR

########################################
# 1) Full baseline (40 steps, interval 1)
########################################
echo "===== [1] Full baseline (40 steps) ====="
for CLS in "${CLASSES[@]}"; do
  echo "==> class $CLS | full (40 steps)"
  python sample.py \
    --ddim-sample \
    --num-sampling-steps 40 \
    --interval 1 \
    --max-order 1 \
    --sample_dir $OUTDIR \
    --seed 0 \
    --class-labels $CLS
  mv sample_interp/40step2.png $OUTDIR/40step_full_class${CLS}.png
done

########################################
# 2) TaylorSeer (interval = 4)
########################################
echo "===== [2] TaylorSeer (interval 4) ====="
for CLS in "${CLASSES[@]}"; do
  echo "==> class $CLS | TaylorSeer (interval 4)"
  python sample.py \
    --ddim-sample \
    --num-sampling-steps 40 \
    --interval 4 \
    --max-order 1 \
    --sample_dir $OUTDIR \
    --seed 0 \
    --class-labels $CLS
  mv sample_interp/40step2.png $OUTDIR/40step_taylor_class${CLS}.png
done

########################################
# 3) Interpolation (interval = 4 + use_interp)
########################################
echo "===== [3] Interpolation (interval 4, use_interp) ====="
for CLS in "${CLASSES[@]}"; do
  echo "==> class $CLS | Interp (interval 4)"
  python sample.py \
    --ddim-sample \
    --num-sampling-steps 40 \
    --interval 4 \
    --max-order 1 \
    --use_interp \
    --sample_dir $OUTDIR \
    --seed 0 \
    --class-labels $CLS
  mv sample_interp/40step2.png $OUTDIR/40step_interp_class${CLS}.png
done

echo "✅ 모든 실험 완료!"
