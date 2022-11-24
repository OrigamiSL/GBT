python -u main.py --root_path ./data/weather/ --model GBT --data weather --features S --seq_len 96 --label_len 96 --pred_len 96 --s_layers 3,2,1 --d_layers 2 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.05 --d_model 512 --criterion Standard --time --test_inverse

python -u main.py --root_path ./data/weather/ --model GBT --data weather --features S --seq_len 96 --label_len 96 --pred_len 192 --s_layers 3,2,1 --d_layers 2 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.05 --d_model 512 --criterion Standard --time --test_inverse

python -u main.py --root_path ./data/weather/ --model GBT --data weather --features S --seq_len 96 --label_len 96 --pred_len 336 --s_layers 3,2,1 --d_layers 2 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.05 --d_model 512 --criterion Standard --time --test_inverse

python -u main.py --root_path ./data/weather/ --model GBT --data weather --features S --seq_len 96 --label_len 96 --pred_len 720 --s_layers 3,2,1 --d_layers 2 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.05 --d_model 512 --criterion Standard --time --test_inverse

python -u main.py --root_path ./data/weather/ --model GBT --data weather --features M --seq_len 96 --label_len 96 --pred_len 96 --s_layers 3,2,1 --d_layers 2 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.05 --fd_model 64 --d_model 512 --criterion Standard --time --instance --test_inverse

python -u main.py --root_path ./data/weather/ --model GBT --data weather --features M --seq_len 96 --label_len 96 --pred_len 192 --s_layers 3,2,1 --d_layers 2 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.05 --fd_model 64 --d_model 512 --criterion Standard --time --instance --test_inverse

python -u main.py --root_path ./data/weather/ --model GBT --data weather --features M --seq_len 96 --label_len 96 --pred_len 336 --s_layers 3,2,1 --d_layers 2 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.05 --fd_model 64 --d_model 512 --criterion Standard --time --instance --test_inverse

python -u main.py --root_path ./data/weather/ --model GBT --data weather --features M --seq_len 96 --label_len 96 --pred_len 720 --s_layers 3,2,1 --d_layers 2 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.05 --fd_model 64 --d_model 512 --criterion Standard --time --instance --test_inverse
