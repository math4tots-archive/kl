#infdef KLCRT_H
#define KLCRT_H

typedef long KLCint;
typedef double KLCfloat;
typedef struct KLCvar KLCvar;
typedef struct KLCtry KLCtry;
typedef struct KLCobj KLCobj;
typedef struct KLCcls KLCcls;
typedef KLCobj* KLCalloc();
typedef void KLCdelete();
typedef KLCtry KLCmethod(int, KLCvar*);

struct KLCvar {
  int tag;
  union {
    KLCint i;
    KLCfloat f;
    KLCobj* o;
  } u;
};

struct KLCtry {
  int success;
  KLCvar value;
};

struct KLCobj {
  size_t refcnt;
  KLCcls* cls;
  KLCobj* next;
  void* weakref; /* TODO: For weak reference in the future */
};

/* General C utilities */
char* KLCcopyStr(const char*);

/* OOP utilities */
KLCcls* KLCnewClass(const char*, KLCalloc*, KLCdelete*);

KLCvar KLCobjToVar(KLCobj*);
KLCvar KLCintToVar(KLCint);
KLCvar KLCfloatToVar(KLCfloat);



#endif/*KLCRT_H*/
