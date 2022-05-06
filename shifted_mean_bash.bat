@echo off
SetLocal

for %%D in (FASHION) do (
    for %%M in (mandera_detect multi_krum bulyan median tr_mean fltrust) do (
        for %%R in (0 1 2 3 4 5 6 7 8 9) do (
            for %%P in (5 10 15 20 25 30) do (
                python shifted_mean_attack.py --dataset %%D --p_workers %%P --def_method %%M --rep_n %%R
            )
        )
    )
)