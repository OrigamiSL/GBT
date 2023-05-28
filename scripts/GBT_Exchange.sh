python -u main.py --root_path ./data/Exchange/ --model GBT --data Exchange --target Singapore --features S --seq_len 30 --label_len 30 --pred_len 96 --s_layers 1 --d_layers 1 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.1 --d_model 128 --time --criterion Standard --test_inverse

python -u main.py --root_path ./data/Exchange/ --model GBT --data Exchange --target Singapore --features S --seq_len 30 --label_len 30 --pred_len 192 --s_layers 1 --d_layers 1 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.1 --d_model 128 --time --criterion Standard --test_inverse

python -u main.py --root_path ./data/Exchange/ --model GBT --data Exchange --target Singapore --features S --seq_len 30 --label_len 30 --pred_len 336 --s_layers 1 --d_layers 1 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.1 --d_model 128 --time --criterion Standard --test_inverse

python -u main.py --root_path ./data/Exchange/ --model GBT --data Exchange --target Singapore --features S --seq_len 30 --label_len 30 --pred_len 720 --s_layers 1 --d_layers 1 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.1 --d_model 128 --time --criterion Standard --test_inverse

python -u main.py --root_path ./data/Exchange/ --model GBT --data Exchange --target Singapore --features M --seq_len 30 --label_len 30 --pred_len 96 --s_layers 1 --d_layers 1 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.1 --d_model 128 --time --instance --criterion Standard --test_inverse

python -u main.py --root_path ./data/Exchange/ --model GBT --data Exchange --target Singapore --features M --seq_len 30 --label_len 30 --pred_len 192 --s_layers 1 --d_layers 1 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.1 --d_model 128 --time --instance --criterion Standard --test_inverse

python -u main.py --root_path ./data/Exchange/ --model GBT --data Exchange --target Singapore --features M --seq_len 30 --label_len 30 --pred_len 336 --s_layers 1 --d_layers 1 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.1 --d_model 128 --time --instance --criterion Standard --test_inverse

python -u main.py --root_path ./data/Exchange/ --model GBT --data Exchange --target Singapore --features M --seq_len 30 --label_len 30 --pred_len 720 --s_layers 1 --d_layers 1 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.1 --d_model 128 --time --instance --criterion Standard --test_inverse
