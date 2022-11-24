python -u main.py --root_path ./data/ILI/ --model GBT --data ILI --features S --seq_len 36 --label_len 18 --pred_len 24 --s_layers 2,1 --auto_d_layers 1 --attn Full --des 'Exp' --itr 3 --learning_rate 0.0001 --dropout 0.05 --d_model 512 --format autoformer --criterion Standard --test_inverse

python -u main.py --root_path ./data/ILI/ --model GBT --data ILI --features S --seq_len 36 --label_len 18 --pred_len 36 --s_layers 2,1 --auto_d_layers 1 --attn Full --des 'Exp' --itr 3 --learning_rate 0.0001 --dropout 0.05 --d_model 512 --format autoformer --criterion Standard --test_inverse

python -u main.py --root_path ./data/ILI/ --model GBT --data ILI --features S --seq_len 36 --label_len 18 --pred_len 48 --s_layers 2,1 --auto_d_layers 1 --attn Full --des 'Exp' --itr 3 --learning_rate 0.0001 --dropout 0.05 --d_model 512 --format autoformer --criterion Standard --test_inverse

python -u main.py --root_path ./data/ILI/ --model GBT --data ILI --features S --seq_len 36 --label_len 18 --pred_len 60 --s_layers 2,1 --auto_d_layers 1 --attn Full --des 'Exp' --itr 3 --learning_rate 0.0001 --dropout 0.05 --d_model 512 --format autoformer --criterion Standard --test_inverse

python -u main.py --root_path ./data/ILI/ --model GBT --data ILI --features M --seq_len 36 --label_len 18 --pred_len 24 --s_layers 2,1 --auto_d_layers 1 --attn Full --des 'Exp' --itr 3 --batch_size 1 --learning_rate 0.0001 --dropout 0.05 --d_model 512 --format autoformer --criterion Standard --instance --test_inverse

python -u main.py --root_path ./data/ILI/ --model GBT --data ILI --features M --seq_len 36 --label_len 18 --pred_len 36 --s_layers 2,1 --auto_d_layers 1 --attn Full --des 'Exp' --itr 3 --batch_size 1 --learning_rate 0.0001 --dropout 0.05 --d_model 512 --format autoformer --criterion Standard --instance --test_inverse

python -u main.py --root_path ./data/ILI/ --model GBT --data ILI --features M --seq_len 36 --label_len 18 --pred_len 48 --s_layers 2,1 --auto_d_layers 1 --attn Full --des 'Exp' --itr 3 --batch_size 1 --learning_rate 0.0001 --dropout 0.05 --d_model 512 --format autoformer --criterion Standard --instance --test_inverse

python -u main.py --root_path ./data/ILI/ --model GBT --data ILI --features M --seq_len 36 --label_len 18 --pred_len 60 --s_layers 2,1 --auto_d_layers 1 --attn Full --des 'Exp' --itr 3 --batch_size 1 --learning_rate 0.0001 --dropout 0.05 --d_model 512 --format autoformer --criterion Standard --instance --test_inverse
