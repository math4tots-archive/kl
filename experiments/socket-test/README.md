I was observing that after opening and closing lots of connections,
the program would wait a while.

I wasn't sure why this was so.
I wanted to rule out inefficiencies in KLC. So
I tried out a pure C version of socket communication.

I was able to repro consistently that at around 16370 connections,
there was a significant wait.

TODO: Figure out why this happens.
