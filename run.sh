pip install -e . \
  && \
python run_annotator_model.py --noise_type rand1 --val_ratio 0.1 \
  --backbone_name=swsl_resnext101_32x16d \
  --n_epoch=200 --lr=0.001556 --dropout=0.1 --label_smoothing=0.1 \
  --fixmatch_coef=1 --fixmatch_softmax_t=1 --fixmatch_threshold=0.6 \
  --recompute_worker_skills=True --start_recompute_skills_after_epoch=25 \
  && \
python run_annotator_model.py --noise_type worst --val_ratio 0.1 \
  --backbone_name=swsl_resnext101_32x16d \
  --n_epoch=200 --lr=0.001556 --dropout=0.1 --label_smoothing=0.1 \
  --fixmatch_coef=1 --fixmatch_softmax_t=1 --fixmatch_threshold=0.6 \
  --recompute_worker_skills=True --start_recompute_skills_after_epoch=25 \
  && \
python run_annotator_model.py --noise_type aggre --val_ratio 0.1 \
  --backbone_name=swsl_resnext101_32x16d \
  --n_epoch=200 --lr=0.001556 --dropout=0.1 --label_smoothing=0.1 \
  --fixmatch_coef=1 --fixmatch_softmax_t=1 --fixmatch_threshold=0.6 \
  --recompute_worker_skills=True --start_recompute_skills_after_epoch=25
