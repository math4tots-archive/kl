#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#define PORT 8080
#define MAXMSG 512

int make_socket() {
  int sock;

  sock = socket(PF_INET, SOCK_STREAM, 0);
  if (sock < 0) {
    perror("socket");
    exit(1);
  }

  return sock;
}

int main(int argc, char** argv) {
  char buffer[MAXMSG];
  int sk;
  size_t i;
  struct sockaddr_in addr;
  strcpy(buffer, "Hello world!");

  for (i = 0; i < 20000; i++) {
    sk = make_socket();
    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    connect(sk, (struct sockaddr*) &addr, sizeof(addr));
    recv(sk, buffer, strlen(buffer) + 1, 0);
    close(sk);
    printf("socket (%d) message = %s\n", sk, buffer);
  }
}
