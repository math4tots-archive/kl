```
gcc -Wall -Werror -Wpedantic -Wno-unused-function --std=c89 src/*.c && \
cp src/kl.c src/kl.cc && \
g++ -Wall -Werror -Wpedantic -Wno-unused-function --std=c++98 src/kl.cc && \
./a.out && \
rm src/kl.cc a.out
```
