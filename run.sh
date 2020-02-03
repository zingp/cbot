nohup python codes/run.py --mode train >logs/0131.log 2>&1 &

## test
python codes/run.py --mode test --restore ckpt/best_checkpoint.pt
r1=8.280254777070065, r5=26.088110403397028, r10=41.50743099787686, mr=20.239384288747345, mrr=0.18594657430815043
testing time: 149.53014159202576

python codes/run.py --mode test --restore ckpt/checkpoint.pt
r1=32.19214437367304, r5=52.839702760084926, r10=64.70276008492569, mr=13.998142250530785, mrr=0.42666690304409
testing time: 149.10809350013733
