DDShow from https://github.com/Kixiron/ddshow

Quick instructions for use:

Install using `cargo install --path third_party/timely/ddshow`.
This should make `ddshow` available on your `PATH`.

Usage:
Run `ddshow` with two free port numbers (`$PORT1`, `$PORT2` here):
```console
$ ddshow --address=127.0.0.1:$PORT1 --differential-address=127.0.0.1:$PORT2 --differential
Waiting for 1 connection on 127.0.0.1:$PORT1
```
Then run your Pathway program with environment variables `TIMELY_WORKER_LOG_ADDR` and
`DIFFERENTIAL_LOG_ADDR` set, like that:
```console
$ TIMELY_WORKER_LOG_ADDR=127.0.0.1:$PORT1 DIFFERENTIAL_LOG_ADDR=127.0.0.1:$PORT2 python main.py
```
After your program finishes wait a bit for `ddshow` to finish.
Then you get a text summary in `report.txt` and an interactive profile viewer in
`dataflow-graph/graph.html`
(hint: you can serve that directory easily using
`python -m http.server --directory dataflow-graph $ANOTHER_PORT`
and access it via ZeroTier in a browser).

Note: because it uses `abomonation`, `ddshow` needs to be compiled with the same Rust compiler and
timely / differential dataflow versions as the code you are profiling. Remember to reinstall it
after updates.
