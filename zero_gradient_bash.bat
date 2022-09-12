@echo off
SetLocal

for %%D in (QMNIST) do (
    for %%M in (None) do (
        for %%R in (1) do (
            for %%P in (30) do (
                python zero_gradient_attack.py --dataset %%D --p_workers %%P --def_method %%M --rep_n %%R
            )
        )
    )
)