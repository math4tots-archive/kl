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

int make_socket(int port) {
  int sock;
  struct sockaddr_in name;

  sock = socket(PF_INET, SOCK_STREAM, 0);
  if (sock < 0) {
    perror("socket");
    exit(1);
  }

  name.sin_family = AF_INET;
  name.sin_port = htons(port);
  name.sin_addr.s_addr = htonl(INADDR_ANY);
  if (bind(sock, (struct sockaddr*) &name, sizeof(name)) < 0) {
    perror("bind");
    exit(1);
  }
  return sock;
}

int main(int argc, char** argv) {
  const char hello[] = "Hello visitor ";
  char buffer[MAXMSG];
  int sk;
  unsigned long count = 0;
  strcpy(buffer, hello);

  sk = make_socket(PORT);
  listen(sk, 10);

  printf("ready\n");
  for (;;) {
    count++;
    sprintf(buffer + strlen(hello), "%lu", count);
    int s = accept(sk, NULL, NULL);
    if (send(s, buffer, strlen(buffer) + 1, 0) < 0) {
      perror("send");
      exit(1);
    }
    close(s);
    printf("data socket (%d) message sent (%s)\n", s, buffer);
  }
}
