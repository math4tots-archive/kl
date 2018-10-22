#(append C_HEADERS 'klc_os_net.h')
#(append C_SOURCES 'klc_os_net.c')

extern int INADDR_ANY
extern int AF_INET
extern int AF_INET6
extern int SOCK_STREAM
extern int SOCK_DGRAM
extern int SOCK_SEQPACKET
extern int IPPROTO_TCP
extern int IPPROTO_UDP

extern class Socket {
  Try tryShutdown()
  Try tryBindIp4(int port, int addr)

  void shutdown() {
    this.tryShutdown().orDie()
  }

  void bindIp4(int port, int addr) {
    this.tryBindIp4(port, addr).orDie()
  }
}

int inet_addr(String addr)
int htonl(int x)
int htons(int x)
int ntohl(int x)
int ntohs(int x)

Try tryOpenSocket(int domain, int type, int protocol)

Socket openSocket(int domain, int type, int protocol) {
  return tryOpenSocket(domain, type, protocol).orDie().(Socket)
}
