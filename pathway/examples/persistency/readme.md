Instructions to reproduce basic persistency:

1. Setup persistency storage via env var and run example:

```
PATHWAY_PERSISTENT_STORAGE=pstorage python example_persistency.py
```

2. In `table.csv`, there will be results of identity run.

3. Ctrl+C

4. Re-run, having persistency storage set.

5. On the output table, you'll see only the latest timestamp entries. 

6. During the run time, you can copy extra files from `identity_inputs_2` to see that new lines appear at the end of `table.csv`