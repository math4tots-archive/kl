#(append C_HEADERS 'klc_os_net.h')
#(append C_SOURCES 'klc_os_net.c')

extern int AF_INET
extern int AF_INET6
extern int SOCK_STREAM
extern int SOCK_DGRAM
extern int SOCK_SEQPACKET
extern int IPPROTO_TCP
extern int IPPROTO_UDP

extern class Socket {
  Try tryShutdown()
  void shutdown() {
    this.tryShutdown().getOrDie()
  }
}

Try tryOpenSocket(int domain, int type, int protocol)

Socket openSocket(int domain, int type, int protocol) {
  return tryOpenSocket(domain, type, protocol).getOrDie().(Socket)
}
