rm -rf results/runs results/logs && mkdir results/runs results/logs


# last
export CUDA_VISIBLE_DEVICES=7; python run.py --model BASE        --learning_rate 4.0e-3 --lamda 0.30 --main_view last --aux_views                            >> results/logs/BASE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=6; python run.py --model ShareBottom --learning_rate 4.0e-3 --lamda 0.30 --main_view last --aux_views first mta linear           >> results/logs/ShareBottom.log 2>&1 &
export CUDA_VISIBLE_DEVICES=5; python run.py --model MMoE        --learning_rate 4.0e-3 --lamda 0.30 --main_view last --aux_views first mta linear           >> results/logs/MMoE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=4; python run.py --model PLE         --learning_rate 4.0e-3 --lamda 0.30 --main_view last --aux_views first mta linear           >> results/logs/PLE.log         2>&1 &
export CUDA_VISIBLE_DEVICES=3; python run.py --model HoME        --learning_rate 4.0e-3 --lamda 0.30 --main_view last --aux_views first mta linear           >> results/logs/HoME.log        2>&1 &
export CUDA_VISIBLE_DEVICES=2; python run.py --model NATAL       --learning_rate 4.0e-3 --lamda 0.30 --main_view last --aux_views first mta linear cartesian >> results/logs/NATAL.log       2>&1 &
wait
export CUDA_VISIBLE_DEVICES=7; python run.py --model MoAE        --learning_rate 4.0e-3 --lamda 0.30 --main_view last --aux_views last                       >> results/logs/MoAE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=6; python run.py --model MoAE        --learning_rate 4.0e-3 --lamda 0.30 --main_view last --aux_views first                      >> results/logs/MoAE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=5; python run.py --model MoAE        --learning_rate 4.0e-3 --lamda 0.30 --main_view last --aux_views mta                        >> results/logs/MoAE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=4; python run.py --model MoAE        --learning_rate 4.0e-3 --lamda 0.30 --main_view last --aux_views linear                     >> results/logs/MoAE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=3; python run.py --model MoAE        --learning_rate 4.0e-3 --lamda 0.30 --main_view last --aux_views first mta linear           >> results/logs/MoAE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=2; python run.py --model MoAE        --learning_rate 4.0e-3 --lamda 0.30 --main_view last --aux_views first mta linear cartesian >> results/logs/MoAE.log        2>&1 &
wait


# first
export CUDA_VISIBLE_DEVICES=7; python run.py --model BASE        --learning_rate 3.0e-3 --lamda 0.10 --main_view first --aux_views                           >> results/logs/BASE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=6; python run.py --model ShareBottom --learning_rate 3.0e-3 --lamda 0.10 --main_view first --aux_views last                      >> results/logs/ShareBottom.log 2>&1 &
export CUDA_VISIBLE_DEVICES=5; python run.py --model MMoE        --learning_rate 3.0e-3 --lamda 0.10 --main_view first --aux_views last                      >> results/logs/MMoE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=4; python run.py --model PLE         --learning_rate 3.0e-3 --lamda 0.10 --main_view first --aux_views last                      >> results/logs/PLE.log         2>&1 &
export CUDA_VISIBLE_DEVICES=3; python run.py --model HoME        --learning_rate 3.0e-3 --lamda 0.10 --main_view first --aux_views last                      >> results/logs/HoME.log        2>&1 &
export CUDA_VISIBLE_DEVICES=2; python run.py --model NATAL       --learning_rate 3.0e-3 --lamda 0.10 --main_view first --aux_views last                      >> results/logs/NATAL.log       2>&1 &
wait
export CUDA_VISIBLE_DEVICES=7; python run.py --model MoAE        --learning_rate 3.0e-3 --lamda 0.10 --main_view first --aux_views first                     >> results/logs/MoAE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=6; python run.py --model MoAE        --learning_rate 3.0e-3 --lamda 0.10 --main_view first --aux_views last                      >> results/logs/MoAE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=5; python run.py --model MoAE        --learning_rate 3.0e-3 --lamda 0.10 --main_view first --aux_views mta                       >> results/logs/MoAE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=4; python run.py --model MoAE        --learning_rate 3.0e-3 --lamda 0.10 --main_view first --aux_views linear                    >> results/logs/MoAE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=3; python run.py --model MoAE        --learning_rate 3.0e-3 --lamda 0.10 --main_view first --aux_views last mta linear           >> results/logs/MoAE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=2; python run.py --model MoAE        --learning_rate 3.0e-3 --lamda 0.10 --main_view first --aux_views last mta linear cartesian >> results/logs/MoAE.log        2>&1 &
wait


# mta
export CUDA_VISIBLE_DEVICES=7; python run.py --model BASE        --learning_rate 3.5e-3 --lamda 0.30 --main_view mta --aux_views                             >> results/logs/BASE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=6; python run.py --model ShareBottom --learning_rate 3.5e-3 --lamda 0.30 --main_view mta --aux_views last first linear           >> results/logs/ShareBottom.log 2>&1 &
export CUDA_VISIBLE_DEVICES=5; python run.py --model MMoE        --learning_rate 3.5e-3 --lamda 0.30 --main_view mta --aux_views last first linear           >> results/logs/MMoE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=4; python run.py --model PLE         --learning_rate 3.5e-3 --lamda 0.30 --main_view mta --aux_views last first linear           >> results/logs/PLE.log         2>&1 &
export CUDA_VISIBLE_DEVICES=3; python run.py --model HoME        --learning_rate 3.5e-3 --lamda 0.30 --main_view mta --aux_views last first linear           >> results/logs/HoME.log        2>&1 &
export CUDA_VISIBLE_DEVICES=2; python run.py --model NATAL       --learning_rate 3.5e-3 --lamda 0.30 --main_view mta --aux_views last first linear cartesian >> results/logs/NATAL.log       2>&1 &
wait
export CUDA_VISIBLE_DEVICES=7; python run.py --model MoAE        --learning_rate 3.5e-3 --lamda 0.30 --main_view mta --aux_views mta                         >> results/logs/MoAE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=6; python run.py --model MoAE        --learning_rate 3.5e-3 --lamda 0.30 --main_view mta --aux_views last                        >> results/logs/MoAE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=5; python run.py --model MoAE        --learning_rate 3.5e-3 --lamda 0.30 --main_view mta --aux_views first                       >> results/logs/MoAE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=4; python run.py --model MoAE        --learning_rate 3.5e-3 --lamda 0.30 --main_view mta --aux_views linear                      >> results/logs/MoAE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=3; python run.py --model MoAE        --learning_rate 3.5e-3 --lamda 0.30 --main_view mta --aux_views last first linear           >> results/logs/MoAE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=2; python run.py --model MoAE        --learning_rate 3.5e-3 --lamda 0.30 --main_view mta --aux_views last first linear cartesian >> results/logs/MoAE.log        2>&1 &
wait


# linear
export CUDA_VISIBLE_DEVICES=7; python run.py --model BASE        --learning_rate 3.5e-3 --lamda 0.20 --main_view linear --aux_views                          >> results/logs/BASE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=6; python run.py --model ShareBottom --learning_rate 3.5e-3 --lamda 0.20 --main_view linear --aux_views last first mta           >> results/logs/ShareBottom.log 2>&1 &
export CUDA_VISIBLE_DEVICES=5; python run.py --model MMoE        --learning_rate 3.5e-3 --lamda 0.20 --main_view linear --aux_views last first mta           >> results/logs/MMoE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=4; python run.py --model PLE         --learning_rate 3.5e-3 --lamda 0.20 --main_view linear --aux_views last first mta           >> results/logs/PLE.log         2>&1 &
export CUDA_VISIBLE_DEVICES=3; python run.py --model HoME        --learning_rate 3.5e-3 --lamda 0.20 --main_view linear --aux_views last first mta           >> results/logs/HoME.log        2>&1 &
export CUDA_VISIBLE_DEVICES=2; python run.py --model NATAL       --learning_rate 3.5e-3 --lamda 0.20 --main_view linear --aux_views last first mta cartesian >> results/logs/NATAL.log       2>&1 &
wait
export CUDA_VISIBLE_DEVICES=7; python run.py --model MoAE        --learning_rate 3.5e-3 --lamda 0.20 --main_view linear --aux_views linear                   >> results/logs/MoAE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=6; python run.py --model MoAE        --learning_rate 3.5e-3 --lamda 0.20 --main_view linear --aux_views last                     >> results/logs/MoAE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=5; python run.py --model MoAE        --learning_rate 3.5e-3 --lamda 0.20 --main_view linear --aux_views first                    >> results/logs/MoAE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=4; python run.py --model MoAE        --learning_rate 3.5e-3 --lamda 0.20 --main_view linear --aux_views mta                      >> results/logs/MoAE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=3; python run.py --model MoAE        --learning_rate 3.5e-3 --lamda 0.20 --main_view linear --aux_views last first mta           >> results/logs/MoAE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=2; python run.py --model MoAE        --learning_rate 3.5e-3 --lamda 0.20 --main_view linear --aux_views last first mta cartesian >> results/logs/MoAE.log        2>&1 &
wait


# GCSGrad
export CUDA_VISIBLE_DEVICES=7; python run.py --model ShareBottom --GCSGrad --learning_rate 4.0e-3 --lamda 0.30 --main_view last --aux_views first mta linear           >> results/logs/ShareBottom.log 2>&1 &
export CUDA_VISIBLE_DEVICES=6; python run.py --model MMoE        --GCSGrad --learning_rate 4.0e-3 --lamda 0.30 --main_view last --aux_views first mta linear           >> results/logs/MMoE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=5; python run.py --model PLE         --GCSGrad --learning_rate 4.0e-3 --lamda 0.30 --main_view last --aux_views first mta linear           >> results/logs/PLE.log         2>&1 &
export CUDA_VISIBLE_DEVICES=4; python run.py --model HoME        --GCSGrad --learning_rate 4.0e-3 --lamda 0.30 --main_view last --aux_views first mta linear           >> results/logs/HoME.log        2>&1 &
export CUDA_VISIBLE_DEVICES=3; python run.py --model NATAL       --GCSGrad --learning_rate 4.0e-3 --lamda 0.30 --main_view last --aux_views first mta linear cartesian >> results/logs/NATAL.log       2>&1 &
export CUDA_VISIBLE_DEVICES=2; python run.py --model MoAE        --GCSGrad --learning_rate 4.0e-3 --lamda 0.30 --main_view last --aux_views first mta linear cartesian >> results/logs/MoAE.log        2>&1 &
wait


# PCGrad
export CUDA_VISIBLE_DEVICES=7; python run.py --model ShareBottom --PCGrad --learning_rate 4.0e-3 --lamda 0.30 --main_view last --aux_views first mta linear           >> results/logs/ShareBottom.log 2>&1 &
export CUDA_VISIBLE_DEVICES=6; python run.py --model MMoE        --PCGrad --learning_rate 4.0e-3 --lamda 0.30 --main_view last --aux_views first mta linear           >> results/logs/MMoE.log        2>&1 &
export CUDA_VISIBLE_DEVICES=5; python run.py --model PLE         --PCGrad --learning_rate 4.0e-3 --lamda 0.30 --main_view last --aux_views first mta linear           >> results/logs/PLE.log         2>&1 &
export CUDA_VISIBLE_DEVICES=4; python run.py --model HoME        --PCGrad --learning_rate 4.0e-3 --lamda 0.30 --main_view last --aux_views first mta linear           >> results/logs/HoME.log        2>&1 &
export CUDA_VISIBLE_DEVICES=3; python run.py --model NATAL       --PCGrad --learning_rate 4.0e-3 --lamda 0.30 --main_view last --aux_views first mta linear cartesian >> results/logs/NATAL.log       2>&1 &
export CUDA_VISIBLE_DEVICES=2; python run.py --model MoAE        --PCGrad --learning_rate 4.0e-3 --lamda 0.30 --main_view last --aux_views first mta linear cartesian >> results/logs/MoAE.log        2>&1 &
wait
