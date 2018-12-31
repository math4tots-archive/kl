#ifndef klc_os_net_h
#define klc_os_net_h

#include "klc_prelude.h"

typedef struct KLCNosZBnetZBSocket KLCNosZBnetZBSocket;

extern KLC_ti KLC_typeosZBnetZBSocket;

KLCNTry* KLCNosZBnetZBtryOpenSocket(KLC_int domain, KLC_int type, KLC_int protocol);
void KLC_deleteosZBnetZBSocket(KLC_header*, KLC_header**);
KLCNTry* KLCNosZBnetZBSocketZFtryBindIp4(KLCNosZBnetZBSocket* sock, KLC_int port, KLC_int addr);
KLC_int KLCNosZBnetZBinetZAaddr(KLCNString* s);
KLCNTry* KLCNosZBnetZBSocketZFtryClose(KLCNosZBnetZBSocket* sock);
KLCNTry* KLCNosZBnetZBSocketZFtrySendBufferWithFlags(KLCNosZBnetZBSocket* sock, KLCNBuffer* buffer, KLC_int flags);
KLCNTry* KLCNosZBnetZBSocketZFtryRecvBufferWithFlags(KLCNosZBnetZBSocket* sock, KLCNBuffer* buffer, KLC_int flags);

#endif/*klc_os_net_h*/
