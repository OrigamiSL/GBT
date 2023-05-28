python -u main.py --root_path ./data/ECL/ --model GBT --data ECL --target MT_321 --features S --seq_len 168 --label_len 168 --pred_len 96 --s_layers 3,2,1 --d_layers 2 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.05 --d_model 512  --criterion Standard --test_inverse

python -u main.py --root_path ./data/ECL/ --model GBT --data ECL --target MT_321 --features S --seq_len 168 --label_len 168 --pred_len 192 --s_layers 3,2,1 --d_layers 2 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.05 --d_model 512  --criterion Standard --test_inverse

python -u main.py --root_path ./data/ECL/ --model GBT --data ECL --target MT_321 --features S --seq_len 168 --label_len 168 --pred_len 336 --s_layers 3,2,1 --d_layers 2 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.05 --d_model 512  --criterion Standard --test_inverse

python -u main.py --root_path ./data/ECL/ --model GBT --data ECL --target MT_321 --features S --seq_len 168 --label_len 168 --pred_len 720 --s_layers 3,2,1 --d_layers 2 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.05 --d_model 512  --criterion Standard --test_inverse

python -u main.py --root_path ./data/ECL/ --model GBT --data ECL --target MT_321 --features M --seq_len 168 --label_len 168 --pred_len 96 --s_layers 3,2,1 --d_layers 1 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.05 --d_model 512 --instance --criterion Standard --test_inverse

python -u main.py --root_path ./data/ECL/ --model GBT --data ECL --target MT_321 --features M --seq_len 168 --label_len 168 --pred_len 192 --s_layers 3,2,1 --d_layers 1 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.05 --d_model 512 --instance --criterion Standard --test_inverse

python -u main.py --root_path ./data/ECL/ --model GBT --data ECL --target MT_321 --features M --seq_len 168 --label_len 168 --pred_len 336 --s_layers 3,2,1 --d_layers 1 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.05 --d_model 512 --instance --criterion Standard --test_inverse

python -u main.py --root_path ./data/ECL/ --model GBT --data ECL --target MT_321 --features M --seq_len 168 --label_len 168 --pred_len 720 --s_layers 3,2,1 --d_layers 1 --attn Full --des 'Exp' --itr 5 --learning_rate 0.0001 --dropout 0.05 --d_model 512 --instance --criterion Standard --test_inverse
