python -u main.py --model GBT --data ETTh1 --features S --seq_len 168 --label_len 168 --pred_len 96 --s_layers 3,2,1 --d_layers 2 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.05 --d_model 512 --time  --criterion Standard --test_inverse

python -u main.py --model GBT --data ETTh1 --features S --seq_len 168 --label_len 168 --pred_len 192 --s_layers 3,2,1 --d_layers 2 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.05 --d_model 512 --time  --criterion Standard --test_inverse

python -u main.py --model GBT --data ETTh1 --features S --seq_len 168 --label_len 168 --pred_len 336 --s_layers 3,2,1 --d_layers 2 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.05 --d_model 512 --time  --criterion Standard --test_inverse

python -u main.py --model GBT --data ETTh1 --features S --seq_len 168 --label_len 168 --pred_len 720 --s_layers 3,2,1 --d_layers 2 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.05 --d_model 512 --time  --criterion Standard --test_inverse

python -u main.py --model GBT --data ETTh1 --features M --seq_len 168 --label_len 168 --pred_len 96 --s_layers 3,2,1 --d_layers 2 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.1 --fd_model 16 --d_model 512 --time --instance --criterion Standard --test_inverse

python -u main.py --model GBT --data ETTh1 --features M --seq_len 168 --label_len 168 --pred_len 192 --s_layers 3,2,1 --d_layers 2 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.1 --fd_model 16 --d_model 512 --time --instance --criterion Standard --test_inverse

python -u main.py --model GBT --data ETTh1 --features M --seq_len 168 --label_len 168 --pred_len 336 --s_layers 3,2,1 --d_layers 2 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.1 --fd_model 16 --d_model 512 --time --instance --batch_size 16 --criterion Standard --test_inverse

python -u main.py --model GBT --data ETTh1 --features M --seq_len 168 --label_len 168 --pred_len 720 --s_layers 3,2,1 --d_layers 2 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.1 --fd_model 16  --d_model 512 --time --instance --batch_size 8 --criterion Standard --test_inverse

python -u main.py --model GBT --data ETTm2 --features S --seq_len 168 --label_len 168 --pred_len 96 --s_layers 3,2,1 --d_layers 2 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.05 --d_model 512 --time  --criterion Standard --test_inverse

python -u main.py --model GBT --data ETTm2 --features S --seq_len 168 --label_len 168 --pred_len 192 --s_layers 3,2,1 --d_layers 2 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.05 --d_model 512 --time  --criterion Standard --test_inverse

python -u main.py --model GBT --data ETTm2 --features S --seq_len 168 --label_len 168 --pred_len 336 --s_layers 3,2,1 --d_layers 2 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.05 --d_model 512 --time  --criterion Standard --test_inverse

python -u main.py --model GBT --data ETTm2 --features S --seq_len 168 --label_len 168 --pred_len 720 --s_layers 3,2,1 --d_layers 2 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.05 --d_model 512 --time  --criterion Standard --test_inverse

python -u main.py --model GBT --data ETTm2 --features M --seq_len 672 --label_len 672 --pred_len 96 --s_layers 3,2,1 --d_layers 2 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.1 --fd_model 16 --d_model 512 --time --instance  --criterion Standard --test_inverse

python -u main.py --model GBT --data ETTm2 --features M --seq_len 672 --label_len 672 --pred_len 192 --s_layers 3,2,1 --d_layers 2 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.1 --fd_model 16 --d_model 512 --time --instance --criterion Standard --test_inverse

python -u main.py --model GBT --data ETTm2 --features M --seq_len 672 --label_len 672 --pred_len 336 --s_layers 3,2,1 --d_layers 2 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.1  --fd_model 16 --d_model 128 --time --instance  --batch_size 16 --criterion Standard --test_inverse

python -u main.py --model GBT --data ETTm2 --features M --seq_len 672 --label_len 672 --pred_len 720 --s_layers 3,2,1 --d_layers 2 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.1  --fd_model 16 --d_model 128 --time --instance  --batch_size 16 --criterion Standard --test_inverse
