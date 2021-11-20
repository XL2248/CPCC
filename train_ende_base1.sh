code_dir=thumt1_code
work_dir=$PWD
data_dir=path_to_data_file
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
python $work_dir/$code_dir/thumt/bin/trainer.py \
  --model transformer \
  --output $work_dir/model_baseline_en_de_8gpu_uc1_dp.1 \
  --input $data_dir/en2de.en.nodup.norm.tok.clean.bpe $data_dir/en2de.de.nodup.norm.tok.clean.bpe \
  --vocabulary $data_dir/ende.bpe32k.vocab4.txt $data_dir/ende.bpe32k.vocab4.txt \
  --validation $data_dir/newstest2019-ende-src.en.tok.bpe \
  --references $data_dir/newstest2019-ende-ref.de.tok \
  --parameters=device_list=[0,1,2,3,4,5,6,7],update_cycle=1,eval_steps=5000,train_steps=200000,batch_size=4096,max_length=128,constant_batch_size=False,residual_dropout=0.1,attention_dropout=0.1,relu_dropout=0.1,shared_source_target_embedding=True