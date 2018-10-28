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

// https://stackoverflow.com/questions/6729366/what-is-the-difference-between-af-inet-and-pf-inet-in-socket-programming
int PF_INET = AF_INET
int PF_INET6 = AF_INET6

extern class Socket {
  Try tryShutdown()
  Try tryBindIp4(int addr, int port)
  Try tryClose()
  Try tryListen(int backlog)
  Try tryConnectIp4(int addr, int port)
  Try tryAcceptIp4()

  void shutdown() {
    this.tryShutdown().orDie()
  }

  void bindIp4(int addr, int port) {
    this.tryBindIp4(addr, port).orDie()
  }

  void close() {
    this.tryClose().orDie()
  }

  void listen(int backlog) {
    this.tryListen(backlog).orDie()
  }

  void connectIp4(int addr, int port) {
    this.tryConnectIp4(addr, port).orDie()
  }

  List acceptIp4() {
    return this.tryAcceptIp4().orDie().(List)
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
