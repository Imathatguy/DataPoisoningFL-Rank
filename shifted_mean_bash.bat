@echo off
SetLocal

for %%D in (FASHION) do (
    for %%M in (fltrust) do (
        for %%R in (0) do (
            for %%P in (5) do (
                python shifted_mean_attack.py --dataset %%D --p_workers %%P --def_method %%M --rep_n %%R
            )
        )
    )
)