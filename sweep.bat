python run.py --src_data=3 --tgt_data=2
python run.py --src_data=2 --tgt_data=1
python run.py --src_data=1 --tgt_data=0
python run.py --src_data=0 --tgt_data=3
python run.py --src_data=3 --tgt_data=2

python run.py --src_data=3 --tgt_data=2 --model_name_or_path=resmlp_12_distilled_224 --embed_dim=1000
python run.py --src_data=2 --tgt_data=1 --model_name_or_path=resmlp_12_distilled_224 --embed_dim=1000
python run.py --src_data=1 --tgt_data=0 --model_name_or_path=resmlp_12_distilled_224 --embed_dim=1000
python run.py --src_data=0 --tgt_data=3 --model_name_or_path=resmlp_12_distilled_224 --embed_dim=1000