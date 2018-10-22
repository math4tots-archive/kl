#include "klc_os_net.h"

#if KLC_POSIX
#include <errno.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#endif

struct KLCNosZBnetZBSocket {
  KLC_header header;
  int domain;
  int type;
  int protocol;
  int socket;
};

KLCNTry* KLCNosZBnetZBtryOpenSocket(KLC_int domain, KLC_int type, KLC_int protocol) {
  #if KLC_POSIX
    KLCNTry* t;
    KLCNosZBnetZBSocket* sock;
    int s = socket((int) domain, (int) type, (int) protocol);
    if (s == -1) {
      int errval = errno;
      return KLC_failm(strerror(errval));
    }
    sock = (KLCNosZBnetZBSocket*) malloc(sizeof(KLCNosZBnetZBSocket));
    KLC_init_header(&sock->header, &KLC_typeosZBnetZBSocket);
    sock->domain = domain;
    sock->type = type;
    sock->protocol = protocol;
    sock->socket = s;
    t = KLCNTryZEnew(1, KLC_object_to_var((KLC_header*) sock));
    KLC_release((KLC_header*) sock);
    return t;
  #else
    return NULL;
  #endif
}

KLCNTry* KLCNosZBnetZBSocketZFtryShutdown(KLCNosZBnetZBSocket* sock) {
  #if KLC_POSIX
    if (shutdown(sock->socket, SHUT_RDWR) != 0) {
      int errval = errno;
      return KLC_failm(strerror(errval));
    } else {
      return KLCNTryZEnew(1, KLC_int_to_var(0));
    }
  #else
    return NULL;
  #endif
}

void KLC_deleteosZBnetZBSocket(KLC_header* robj, KLC_header** dq) {
  #if KLC_POSIX
    KLCNosZBnetZBSocket* s = (KLCNosZBnetZBSocket*) robj;

    /* TODO: Figure out what to do if shutdown fails */
    shutdown(s->socket, SHUT_RDWR);
  #endif
}

KLC_int KLCNosZBnetZBAFZAINET6ZEinit() {
  #if KLC_POSIX
    return AF_INET6;
  #else
    return -1;
  #endif
}

KLC_int KLCNosZBnetZBAFZAINETZEinit() {
  #if KLC_POSIX
    return AF_INET;
  #else
    return -1;
  #endif
}

KLC_int KLCNosZBnetZBIPPROTOZATCPZEinit() {
  #if KLC_POSIX
    return IPPROTO_TCP;
  #else
    return -1;
  #endif
}

KLC_int KLCNosZBnetZBIPPROTOZAUDPZEinit() {
  #if KLC_POSIX
    return IPPROTO_UDP;
  #else
    return -1;
  #endif
}

KLC_int KLCNosZBnetZBSOCKZADGRAMZEinit() {
  #if KLC_POSIX
    return SOCK_DGRAM;
  #else
    return -1;
  #endif
}

KLC_int KLCNosZBnetZBSOCKZASEQPACKETZEinit() {
  #if KLC_POSIX
    return SOCK_SEQPACKET;
  #else
    return -1;
  #endif
}

KLC_int KLCNosZBnetZBSOCKZASTREAMZEinit() {
  #if KLC_POSIX
    return SOCK_STREAM;
  #else
    return -1;
  #endif
}
