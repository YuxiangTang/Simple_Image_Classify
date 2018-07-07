@echo off
set /a sum=0
setlocal enabledelayedexpansion
for  %%x in (*) do (
    if not "%%x"=="demo.bat" (
        
        rename "%%x" "3_rabbit_!sum!.jpg"
	set /a sum+=1
     
)
)
pause