# MultiWOZ Evaluation

This directory contains the evaluation for MultiWOZ, copied exactly from 
[Tomiinek/MultiWOZ_Evaluation](https://github.com/Tomiinek/MultiWOZ_Evaluation) with the following modifications:

1. Making `time_str_to_minutes` robust to invalid input:

Before:
```python
def time_str_to_minutes(time_string):
    if not re.match(r"[0-9][0-9]:[0-9][0-9]", time_string):
        return 0
    return int(time_string.split(':')[0]) * 60 + int(time_string.split(':')[1])
```
After:
```python
def time_str_to_minutes(time_string):
    time_string = time_string.strip()
    if not re.match(r"^[0-9]?[0-9]:[0-9][0-9]$", time_string):
        return 0
    return int(time_string.split(':')[0]) * 60 + int(time_string.split(':')[1])
```
- Invalid time stamps that include a time within them will now return 0 instead of throwing an error, matching behavior
for invalid time stamps that do not include a time.
- The regex now accepts times in both "hh:mm" and "h:mm" formats. 