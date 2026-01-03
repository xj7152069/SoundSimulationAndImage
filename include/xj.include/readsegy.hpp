#ifndef READSEGY_HPP
#define READSEGY_HPP

#include "../xjc.h"

float ibm2ieee (float fromf);
void ibm_to_float(unsigned int from[], unsigned int to[], int n, int endian);
void segyhead_endianget(segyhead & head);
void segyhead_initialize(segyhead & head);
template <typename T1> inline void endianchange(T1 & a);
template <typename T1> inline float getendianchange(T1 a);
inline float floatendianchange(float a);
void segyhead_open(segyhead & head, bool sufile);
void fmatendianchange(fmat & data,int n1,int n2);
fmat segyhead_readonetrace_tofmat(segyhead & head, fmat & trace);
//void segyhead_readoneline_tofmat(segyhead & head, fmat & data);

///////////////////////////////////////////////////////////////////
int readSegyOneTraceLittleIEEE(segyhead & head, fmat & trace){
    head.infile.read((char *)(&head.head2), sizeof(head.head2));
    int nz=int(head.head2.ns);
    if(trace.n_rows!=nz){
        trace.zeros(nz,1);
        head.nz=nz;
    }
    float *float_p;
    unsigned int IBMFloatBytes;
    for(int i=0;i<trace.n_rows;i++){
        head.infile.read((char *)&IBMFloatBytes, 4); 
        float_p=(float *)(&IBMFloatBytes);
        trace(i,0)=*float_p;
    }
    return 0;
}
int readSegyOneTraceBigIEEE(segyhead & head, fmat & trace){
    head.infile.read((char *)(&head.head2), sizeof(head.head2));
    int nz=int(getendianchange(head.head2.ns));
    if(trace.n_rows!=nz){
        trace.zeros(nz,1);
        head.nz=nz;
    }
    float *float_p;
    unsigned int IBMFloatBytes;
    for(int i=0;i<trace.n_rows;i++){
        head.infile.read((char *)&IBMFloatBytes, 4); 
        float_p=(float *)(&IBMFloatBytes);
        IBMFloatBytes=getendianchange(IBMFloatBytes);
        trace(i,0)=*float_p;
    }
    return 0;
}
int readSegyOneTraceBigIBM(segyhead & head, fmat & trace){
    head.infile.read((char *)(&head.head2), sizeof(head.head2));
    int nz=int(getendianchange(head.head2.ns));
    if(trace.n_rows!=nz){
        trace.zeros(nz,1);
        head.nz=nz;
    }
    float *float_p;
    unsigned int *int_p;
    float IBMFloatBytes;
    float_p=(float *)(&IBMFloatBytes);
    int_p=(unsigned int *)(&IBMFloatBytes);
    for(int i=0;i<trace.n_rows;i++){
        head.infile.read((char *)&IBMFloatBytes, 4); 
        IBMFloatBytes=getendianchange(IBMFloatBytes);
        ibm_to_float(int_p, int_p, 1, 1);
        trace(i,0)=*float_p;
        if(isinf(abs(trace(i,0)))){
            trace(i,0)=0;
        }
    }
    return 0;
}
int readSegyOneTraceLittleIBM(segyhead & head, fmat & trace){
    head.infile.read((char *)(&head.head2), sizeof(head.head2));
    int nz=int(head.head2.ns);
    if(trace.n_rows!=nz){
        trace.zeros(nz,1);
        head.nz=nz;
    }
    float *float_p;
    unsigned int *int_p;
    float IBMFloatBytes;
    float_p=(float *)(&IBMFloatBytes);
    int_p=(unsigned int *)(&IBMFloatBytes);
    for(int i=0;i<trace.n_rows;i++){
        head.infile.read((char *)&IBMFloatBytes, 4); 
        ibm_to_float(int_p, int_p, 1, 1);
        trace(i,0)=*float_p;
        if(isinf(abs(trace(i,0)))){
            trace(i,0)=0;
        }
    }
    return 0;
}
fmat segyhead_readonetrace_tofmat(segyhead & head, fmat & trace)
{
    int nz;
    head.infile.read((char *)(&head.head2), sizeof(head.head2));
    if(head.endian=='b')
        nz=int(getendianchange(head.head2.ns));
    else
        nz=int(head.head2.ns);
    trace.zeros(nz,1);
    head.nz=nz;

    //cout<<nz<<endl;
    fmat rawtrace;
    rawtrace.copy_size(trace);

    if(!head.isibm && head.endian=='l'){
        int i,j;
        float *float_p;
        unsigned int IBMFloatBytes;
        for(i=0;i<trace.n_rows;i++){
            for(j=0;j<trace.n_cols;j++){
                head.infile.read((char *)&IBMFloatBytes, 4); 
                float_p=(float *)(&IBMFloatBytes);
                rawtrace(i,j)=*float_p;
                trace(i,j)=*float_p;
            }
        }
    }
    else if(!head.isibm && head.endian=='b'){
        int i,j;
        float *float_p;
        unsigned int IBMFloatBytes;
        for(i=0;i<trace.n_rows;i++){
            for(j=0;j<trace.n_cols;j++){
                head.infile.read((char *)&IBMFloatBytes, 4); 
                float_p=(float *)(&IBMFloatBytes);
                rawtrace(i,j)=*float_p;
                IBMFloatBytes=getendianchange(IBMFloatBytes);
                float_p=(float *)(&IBMFloatBytes);
                trace(i,j)=*float_p;
            }
        }
    }
    else if(head.isibm && head.endian=='b'){
        int i,j;
        float *float_p;
        unsigned int *int_p;
        float IBMFloatBytes;
        float_p=(float *)(&IBMFloatBytes);
        int_p=(unsigned int *)(&IBMFloatBytes);
        for(i=0;i<trace.n_rows;i++){
            for(j=0;j<trace.n_cols;j++){
                head.infile.read((char *)&IBMFloatBytes, 4); 
                rawtrace(i,j)=*float_p;
                IBMFloatBytes=getendianchange(IBMFloatBytes);
                ibm_to_float(int_p, int_p, 1, 1);
                trace(i,j)=*float_p;
                if(isinf(abs(trace(i,j)))){
                    trace(i,j)=0;
                }
            }
        }
            //cout<<i<<"ok"<<IBMFloatBytes<<endl;

    }
    else if(head.isibm && head.endian=='l'){
        int i,j;
        float *float_p;
        unsigned int *int_p;
        float IBMFloatBytes;
        float_p=(float *)(&IBMFloatBytes);
        int_p=(unsigned int *)(&IBMFloatBytes);
        for(i=0;i<trace.n_rows;i++){
            for(j=0;j<trace.n_cols;j++){
                head.infile.read((char *)&IBMFloatBytes, 4); 
                rawtrace(i,j)=*float_p;
                ibm_to_float(int_p, int_p, 1, 1);
                trace(i,j)=*float_p;
                if(isinf(abs(trace(i,j)))){
                    trace(i,j)=0;
                }
            }
        }
    }

    return rawtrace;
}
bool ifstreamFloatEndOfFile(ifstream & infile)
{
    float readtest;
    bool endOfFile(false);
    infile.read((char *)(&readtest), sizeof(readtest));
    if(!infile.is_open()){endOfFile=true;cout<<"File has been closed!"<<endl;}
    else if(infile.eof()){endOfFile=true;}
    else{infile.seekg(-sizeof(readtest),ios::cur);}
    return endOfFile;
}
void segyhead_open(segyhead & head, bool sufile=false);
void segyhead_open(segyhead & head, bool sufile)
{
    head.infile.open(head.filename,ios::binary|ios::in);
    head.infile.seekg(0,ios::beg);
    if(!head.infile){cout<<"file open error: "<<head.filename<<endl;}
    if(!sufile){
        head.infile.read((char *)(&head.head0), sizeof(head.head0));
        head.infile.read((char *)(&head.head1), sizeof(head.head1));
        segyhead_endianget(head);
    }else{
        //cout<<"This is a .su file: "<<head.filename<<endl;
        head.endian='l';
        head.isibm=false;
    }
}


void segyhead_initialize(segyhead & head)
{
    head.filename[0] = '\0';
    head.endian = '0';
}

void segyhead_endianget(segyhead & head)
{
    if ((head.head1.format == 1) || (head.head1.format == 2) || \
        (head.head1.format == 3) || (head.head1.format == 4) || \
        (head.head1.format == 5) || (head.head1.format == 6) || \
        (head.head1.format == 7) || (head.head1.format == 8))
    {
        head.endian = 'l'; //小端，高位字节在前，低位字节在后
        cout<<"The format of the SEGY-data is little-Endian: "<<\
            head.head1.format<<endl;
        if(head.head1.format == 1)
        {
            head.isibm=true;
            cout<<"The format of the SEGY-data is IBM: "<<\
                head.head1.format<<endl;
        }
        else
        {
            head.isibm=false;
            cout<<"The format of the SEGY-data is IEEE: "<<\
                head.head1.format<<endl;
        }
    }
    else
    {
        short swapNum=head.head1.format;
        endianchange(swapNum);
        if ((swapNum == 1) || (swapNum == 2) || \
            (swapNum == 3) || (swapNum == 4) || \
            (swapNum == 5) || (swapNum == 6) || \
            (swapNum == 7) || (swapNum == 8))
        {
            head.endian = 'b'; //大端(654321)，与小端相反，需要调整为小端
            cout<<"The format of the SEGY-data is Big-Endian: "<<\
                swapNum<<endl;
            if(swapNum == 1)
            {
                head.isibm=true;
                cout<<"The format of the SEGY-data is IBM: "<<\
                    swapNum<<endl;
            }
            else
            {
                head.isibm=false;
                cout<<"The format of the SEGY-data is IEEE: "<<\
                    swapNum<<endl;
            }
        }else
        {
            head.endian = '0';
            cout<<"Error:the format of the SEGY-data is not know:"<<swapNum<<endl;
        }
    } 
    
}

template <typename T1> 
inline void endianchange(T1 & a)
{
    char *p,t;
    int i,bit_num;
    bit_num=sizeof(a);
    p=(char *)(&a);
    for(i=0;i<bit_num/2;i++)
    {
        t=*(p+i);
        *(p+i)=*(p+bit_num-i-1);
        *(p+bit_num-i-1)=t;
    }
}

void fmatendianchange(fmat & data,int n1,int n2)
{
    int i,j;
    float a;
    for(i=0;i<n1;i++)
    {
        for(j=0;j<n2;j++)
        {
            a=data(i,j);
            data(i,j)=floatendianchange(a);
        }
    }
}

inline float floatendianchange(float a)
{
    typedef union SWAP_UNION{
        float f;
        char  c[4];
        }SWAP_UNION;
    SWAP_UNION d1,d2;
    d1.f=a;
    d2.c[0]=d1.c[3];
    d2.c[1]=d1.c[2];
    d2.c[2]=d1.c[1];
    d2.c[3]=d1.c[0];
    return d2.f;
}

template <typename T1> 
inline float getendianchange(T1 a)
{
    char *p,t;
    int i,bit_num;
    bit_num=sizeof(a);
    p=(char *)(&a);
    for(i=0;i<bit_num/2;i++)
    {
        t=*(p+i);
        *(p+i)=*(p+bit_num-i-1);
        *(p+bit_num-i-1)=t;
    }
    float b;
    b=float(a);
    return b;
}
segyhead1 changeSegyHeadEndian(segyhead1 hdr){
    hdr.bgrcv=getendianchange(hdr.bgrcv);
    hdr.dto=getendianchange(hdr.dto);
    hdr.fold=getendianchange(hdr.fold);
    hdr.format=getendianchange(hdr.format);
    hdr.hcorr=getendianchange(hdr.hcorr);
    hdr.hdt=getendianchange(hdr.hdt);
    hdr.hns=getendianchange(hdr.hns);
    hdr.hsfe=getendianchange(hdr.hsfe);
    hdr.hsfs=getendianchange(hdr.hsfs);
    hdr.hslen=getendianchange(hdr.hslen);
    hdr.hstae=getendianchange(hdr.hstae);
    hdr.hstas=getendianchange(hdr.hstas);
    hdr.hstyp=getendianchange(hdr.hstyp);
    hdr.htatyp=getendianchange(hdr.htatyp);
    for(int k=0;k<170;k++){
        hdr.hunass[k]=getendianchange(hdr.hunass[k]);
    }
    hdr.jobid=getendianchange(hdr.jobid);
    hdr.lino=getendianchange(hdr.lino);
    hdr.mfeet=getendianchange(hdr.mfeet);
    hdr.nart=getendianchange(hdr.nart);
    hdr.nso=getendianchange(hdr.nso);
    hdr.ntrpr=getendianchange(hdr.ntrpr);
    hdr.polyt=getendianchange(hdr.polyt);
    hdr.rcvm=getendianchange(hdr.rcvm);
    hdr.reno=getendianchange(hdr.reno);
    hdr.schn=getendianchange(hdr.schn);
    hdr.tsort=getendianchange(hdr.tsort);
    hdr.vpol=getendianchange(hdr.vpol);
    hdr.vscode=getendianchange(hdr.vscode);
    return hdr;
}
segyhead2 changeSuHeadEndian(segyhead2 hdr){
    hdr.afilf=getendianchange(hdr.afilf);
    hdr.afils=getendianchange(hdr.afils);
    hdr.cdp=getendianchange(hdr.cdp);
    hdr.cdpt=getendianchange(hdr.cdpt);
    hdr.corr=getendianchange(hdr.corr);
    hdr.counit=getendianchange(hdr.counit);
    hdr.d1=getendianchange(hdr.d1);
    hdr.day=getendianchange(hdr.day);
    hdr.delrt=getendianchange(hdr.delrt);
    hdr.dt=getendianchange(hdr.dt);
    hdr.duse=getendianchange(hdr.duse);
    hdr.ep=getendianchange(hdr.ep);
    hdr.f1=getendianchange(hdr.f1);
    hdr.fldr=getendianchange(hdr.fldr);
    hdr.gain=getendianchange(hdr.gain);
    hdr.gaps=getendianchange(hdr.gaps);
    hdr.gdel=getendianchange(hdr.gdel);
    hdr.gelev=getendianchange(hdr.gelev);
    hdr.grnlof=getendianchange(hdr.grnlof);
    hdr.grnofr=getendianchange(hdr.grnofr);
    hdr.grnors=getendianchange(hdr.grnors);
    hdr.gstat=getendianchange(hdr.gstat);
    hdr.gut=getendianchange(hdr.gut);
    hdr.gwdep=getendianchange(hdr.gwdep);
    hdr.gx=getendianchange(hdr.gx);
    hdr.gy=getendianchange(hdr.gy);
    hdr.hcf=getendianchange(hdr.hcf);
    hdr.hcs=getendianchange(hdr.hcs);
    hdr.hour=getendianchange(hdr.hour);
    hdr.igc=getendianchange(hdr.igc);
    hdr.igi=getendianchange(hdr.igi);
    hdr.iline=getendianchange(hdr.iline);
    hdr.laga=getendianchange(hdr.laga);
    hdr.lagb=getendianchange(hdr.lagb);
    hdr.lcf=getendianchange(hdr.lcf);
    hdr.lcs=getendianchange(hdr.lcs);
    hdr.mark=getendianchange(hdr.mark);
    hdr.minute=getendianchange(hdr.minute);
    hdr.mute=getendianchange(hdr.mute);
    hdr.muts=getendianchange(hdr.muts);
    hdr.nhs=getendianchange(hdr.nhs);
    hdr.nofilf=getendianchange(hdr.nofilf);
    hdr.nofils=getendianchange(hdr.nofils);
    hdr.ns=getendianchange(hdr.ns);
    hdr.ntr=getendianchange(hdr.ntr);
    hdr.nvs=getendianchange(hdr.nvs);
    hdr.offset=getendianchange(hdr.offset);
    hdr.otrav=getendianchange(hdr.otrav);
    hdr.scalco=getendianchange(hdr.scalco);
    hdr.scalel=getendianchange(hdr.scalel);
    hdr.sdel=getendianchange(hdr.sdel);
    hdr.sdepth=getendianchange(hdr.sdepth);
    hdr.sec=getendianchange(hdr.sec);
    hdr.selev=getendianchange(hdr.selev);
    hdr.sfe=getendianchange(hdr.sfe);
    hdr.sfs=getendianchange(hdr.sfs);
    hdr.shortpad=getendianchange(hdr.shortpad);
    hdr.slen=getendianchange(hdr.slen);
    hdr.sstat=getendianchange(hdr.sstat);
    hdr.stae=getendianchange(hdr.stae);
    hdr.stas=getendianchange(hdr.stas);
    hdr.styp=getendianchange(hdr.styp);
    hdr.sut=getendianchange(hdr.sut);
    hdr.swdep=getendianchange(hdr.swdep);
    hdr.swevel=getendianchange(hdr.swevel);
    hdr.sx=getendianchange(hdr.sx);
    hdr.sy=getendianchange(hdr.sy);
    hdr.tatyp=getendianchange(hdr.tatyp);
    hdr.timbas=getendianchange(hdr.timbas);
    hdr.tracf=getendianchange(hdr.tracf);
    hdr.tracl=getendianchange(hdr.tracl);
    hdr.tracr=getendianchange(hdr.tracr);
    hdr.trid=getendianchange(hdr.trid);
    hdr.trwf=getendianchange(hdr.trwf);
    hdr.tstat=getendianchange(hdr.tstat);
    hdr.ungpow=getendianchange(hdr.ungpow);
    hdr.unscale=getendianchange(hdr.unscale);
    for(int k=0;k<14;k++){
        hdr.unass[k]=getendianchange(hdr.unass[k]);
    }
    hdr.wevel=getendianchange(hdr.wevel);
    hdr.xline=getendianchange(hdr.xline);
    hdr.year=getendianchange(hdr.year);
    return hdr;
}

void zerosSuHead(struct segyhead2 & hdr){
    hdr.afilf=0;
    hdr.afils=0;
    hdr.cdp=0;
    hdr.cdpt=0;
    hdr.corr=0;
    hdr.counit=0;
    hdr.d1=0;
    hdr.day=0;
    hdr.delrt=0;
    hdr.dt=0;
    hdr.duse=0;
    hdr.ep=0;
    hdr.f1=0;
    hdr.fldr=0;
    hdr.gain=0;
    hdr.gaps=0;
    hdr.gdel=0;
    hdr.gelev=0;
    hdr.grnlof=0;
    hdr.grnofr=0;
    hdr.grnors=0;
    hdr.gstat=0;
    hdr.gut=0;
    hdr.gwdep=0;
    hdr.gx=0;
    hdr.gy=0;
    hdr.hcf=0;
    hdr.hcs=0;
    hdr.hour=0;
    hdr.igc=0;
    hdr.igi=0;
    hdr.iline=0;
    hdr.laga=0;
    hdr.lagb=0;
    hdr.lcf=0;
    hdr.lcs=0;
    hdr.mark=0;
    hdr.minute=0;
    hdr.mute=0;
    hdr.muts=0;
    hdr.nhs=0;
    hdr.nofilf=0;
    hdr.nofils=0;
    hdr.ns=0;
    hdr.ntr=0;
    hdr.nvs=0;
    hdr.offset=0;
    hdr.otrav=0;
    hdr.scalco=0;
    hdr.scalel=0;
    hdr.sdel=0;
    hdr.sdepth=0;
    hdr.sec=0;
    hdr.selev=0;
    hdr.sfe=0;
    hdr.sfs=0;
    hdr.shortpad=0;
    hdr.slen=0;
    hdr.sstat=0;
    hdr.stae=0;
    hdr.tatyp=0;
    hdr.timbas=0;
    hdr.tracf=0;
    hdr.tracl=0;
    hdr.tracr=0;
    hdr.trid=0;
    hdr.trwf=0;
    hdr.tstat=0;
    hdr.ungpow=0;
    hdr.unscale=0;
    hdr.wevel=0;
    hdr.xline=0;
    hdr.year=0;
}
float getSegyHeadKey(segyhead &hdr, const char *key)
{
    float p;
if(hdr.endian=='b'){
	if(strcmp(key,"tracl"   )==0)       { p=getendianchange(hdr.head2.tracl);}
    else if (strcmp(key,"tracr"   )==0) { p=getendianchange(hdr.head2.tracr);}
    else if (strcmp(key,"fldr"    )==0) { p=getendianchange(hdr.head2.fldr) ;}
    else if (strcmp(key,"tracf"   )==0) { p=getendianchange(hdr.head2.tracf);}
    else if (strcmp(key,"ep"      )==0) { p=getendianchange(hdr.head2.ep)  ;}
    else if (strcmp(key,"cdp"     )==0) { p=getendianchange(hdr.head2.cdp) ;}
    else if (strcmp(key,"cdpt"    )==0) { p=getendianchange(hdr.head2.cdpt);}
    else if (strcmp(key,"trid"    )==0) { p=getendianchange(hdr.head2.trid);}
    else if (strcmp(key,"nvs"     )==0) { p=getendianchange(hdr.head2.nvs) ;}
    else if (strcmp(key,"nhs"     )==0) { p=getendianchange(hdr.head2.nhs   );}
    else if (strcmp(key,"duse"    )==0) { p=getendianchange(hdr.head2.duse  );}
    else if (strcmp(key,"offset"  )==0) { p=getendianchange(hdr.head2.offset);}
    else if (strcmp(key,"gelev"   )==0) { p=getendianchange(hdr.head2.gelev );}
    else if (strcmp(key,"selev"   )==0) { p=getendianchange(hdr.head2.selev );}
    else if (strcmp(key,"sdepth"  )==0) { p=getendianchange(hdr.head2.sdepth);}
    else if (strcmp(key,"gdel"    )==0) { p=getendianchange(hdr.head2.gdel  );}
    else if (strcmp(key,"sdel"    )==0) { p=getendianchange(hdr.head2.sdel  );}
    else if (strcmp(key,"swdep"   )==0) { p=getendianchange(hdr.head2.swdep );}
    else if (strcmp(key,"gwdep"   )==0) { p=getendianchange(hdr.head2.gwdep );}
    else if (strcmp(key,"scalel"  )==0) { p=getendianchange(hdr.head2.scalel);}
    else if (strcmp(key,"scalco"  )==0) { p=getendianchange(hdr.head2.scalco);}
    else if (strcmp(key,"sx"      )==0) { p=getendianchange(hdr.head2.sx    );}
    else if (strcmp(key,"sy"      )==0) { p=getendianchange(hdr.head2.sy    );}
    else if (strcmp(key,"gx"      )==0) { p=getendianchange(hdr.head2.gx    );}
    else if (strcmp(key,"gy"      )==0) { p=getendianchange(hdr.head2.gy    );}
    else if (strcmp(key,"counit"  )==0) { p=getendianchange(hdr.head2.counit);}
    else if (strcmp(key,"wevel"   )==0) { p=getendianchange(hdr.head2.wevel );}
    else if (strcmp(key,"swevel"  )==0) { p=getendianchange(hdr.head2.swevel);}
    else if (strcmp(key,"sut"     )==0) { p=getendianchange(hdr.head2.sut   );}
    else if (strcmp(key,"gut"     )==0) { p=getendianchange(hdr.head2.gut   );}
    else if (strcmp(key,"sstat"   )==0) { p=getendianchange(hdr.head2.sstat );}
    else if (strcmp(key,"gstat"   )==0) { p=getendianchange(hdr.head2.gstat );}
    else if (strcmp(key,"tstat"   )==0) { p=getendianchange(hdr.head2.tstat );}
    else if (strcmp(key,"laga"    )==0) { p=getendianchange(hdr.head2.laga  );}
    else if (strcmp(key,"lagb"    )==0) { p=getendianchange(hdr.head2.lagb  );}
    else if (strcmp(key,"delrt"   )==0) { p=getendianchange(hdr.head2.delrt );}
    else if (strcmp(key,"muts"    )==0) { p=getendianchange(hdr.head2.muts  );}
    else if (strcmp(key,"mute"    )==0) { p=getendianchange(hdr.head2.mute  );}
    else if (strcmp(key,"ns"      )==0) { p=getendianchange(hdr.head2.ns    );}
    else if (strcmp(key,"dt"      )==0) { p=getendianchange(hdr.head2.dt    );}
    else if (strcmp(key,"gain"    )==0) { p=getendianchange(hdr.head2.gain  );}
    else if (strcmp(key,"igc"     )==0) { p=getendianchange(hdr.head2.igc   );}
    else if (strcmp(key,"igi"     )==0) { p=getendianchange(hdr.head2.igi   );}
    else if (strcmp(key,"corr"    )==0) { p=getendianchange(hdr.head2.corr  );}
    else if (strcmp(key,"sfs"     )==0) { p=getendianchange(hdr.head2.sfs   );}
    else if (strcmp(key,"sfe"     )==0) { p=getendianchange(hdr.head2.sfe   );}
    else if (strcmp(key,"slen"    )==0) { p=getendianchange(hdr.head2.slen  );}
    else if (strcmp(key,"styp"    )==0) { p=getendianchange(hdr.head2.styp  );}
    else if (strcmp(key,"stas"    )==0) { p=getendianchange(hdr.head2.stas  );}
    else if (strcmp(key,"stae"    )==0) { p=getendianchange(hdr.head2.stae  );}
    else if (strcmp(key,"tatyp"   )==0) { p=getendianchange(hdr.head2.tatyp );}
    else if (strcmp(key,"afilf"   )==0) { p=getendianchange(hdr.head2.afilf );}
    else if (strcmp(key,"afils"   )==0) { p=getendianchange(hdr.head2.afils );}
    else if (strcmp(key,"nofilf"  )==0) { p=getendianchange(hdr.head2.nofilf);}
    else if (strcmp(key,"nofils"  )==0) { p=getendianchange(hdr.head2.nofils);}
    else if (strcmp(key,"lcf"     )==0) { p=getendianchange(hdr.head2.lcf   );}
    else if (strcmp(key,"hcf"     )==0) { p=getendianchange(hdr.head2.hcf   );}
    else if (strcmp(key,"lcs"     )==0) { p=getendianchange(hdr.head2.lcs   );}
    else if (strcmp(key,"hcs"     )==0) { p=getendianchange(hdr.head2.hcs   );}
    else if (strcmp(key,"year"    )==0) { p=getendianchange(hdr.head2.year  );}
    else if (strcmp(key,"day"     )==0) { p=getendianchange(hdr.head2.day   );}
    else if (strcmp(key,"hour"    )==0) { p=getendianchange(hdr.head2.hour  );}
    else if (strcmp(key,"minute"  )==0) { p=getendianchange(hdr.head2.minute);}
    else if (strcmp(key,"sec"     )==0) { p=getendianchange(hdr.head2.sec   );}
    else if (strcmp(key,"timbas"  )==0) { p=getendianchange(hdr.head2.timbas);}
    else if (strcmp(key,"trwf"    )==0) { p=getendianchange(hdr.head2.trwf  );}
    else if (strcmp(key,"grnors"  )==0) { p=getendianchange(hdr.head2.grnors);}
    else if (strcmp(key,"grnofr"  )==0) { p=getendianchange(hdr.head2.grnofr);}
    else if (strcmp(key,"grnlof"  )==0) { p=getendianchange(hdr.head2.grnlof);}
    else if (strcmp(key,"gaps"    )==0) { p=getendianchange(hdr.head2.gaps  );}
    else if (strcmp(key,"otrav"   )==0) { p=getendianchange(hdr.head2.otrav );}
    else{p=0;std::cout<<"Error: Not find su key number!!";}
}else{
    if(strcmp(key,"tracl"   )==0)       { p = (hdr.head2.tracl);}
    else if (strcmp(key,"tracr"   )==0) { p = (hdr.head2.tracr);}
    else if (strcmp(key,"fldr"    )==0) { p = (hdr.head2.fldr) ;}
    else if (strcmp(key,"tracf"   )==0) { p = (hdr.head2.tracf);}
    else if (strcmp(key,"ep"      )==0) { p = (hdr.head2.ep)  ;}
    else if (strcmp(key,"cdp"     )==0) { p = (hdr.head2.cdp) ;}
    else if (strcmp(key,"cdpt"    )==0) { p = (hdr.head2.cdpt);}
    else if (strcmp(key,"trid"    )==0) { p = (hdr.head2.trid);}
    else if (strcmp(key,"nvs"     )==0) { p = (hdr.head2.nvs) ;}
    else if (strcmp(key,"nhs"     )==0) { p = hdr.head2.nhs   ;}
    else if (strcmp(key,"duse"    )==0) { p = hdr.head2.duse  ;}
    else if (strcmp(key,"offset"  )==0) { p = hdr.head2.offset;}
    else if (strcmp(key,"gelev"   )==0) { p = hdr.head2.gelev ;}
    else if (strcmp(key,"selev"   )==0) { p = hdr.head2.selev ;}
    else if (strcmp(key,"sdepth"  )==0) { p = hdr.head2.sdepth;}
    else if (strcmp(key,"gdel"    )==0) { p = hdr.head2.gdel  ;}
    else if (strcmp(key,"sdel"    )==0) { p = hdr.head2.sdel  ;}
    else if (strcmp(key,"swdep"   )==0) { p = hdr.head2.swdep ;}
    else if (strcmp(key,"gwdep"   )==0) { p = hdr.head2.gwdep ;}
    else if (strcmp(key,"scalel"  )==0) { p = hdr.head2.scalel;}
    else if (strcmp(key,"scalco"  )==0) { p = hdr.head2.scalco;}
    else if (strcmp(key,"sx"      )==0) { p = hdr.head2.sx    ;}
    else if (strcmp(key,"sy"      )==0) { p = hdr.head2.sy    ;}
    else if (strcmp(key,"gx"      )==0) { p = hdr.head2.gx    ;}
    else if (strcmp(key,"gy"      )==0) { p = hdr.head2.gy    ;}
    else if (strcmp(key,"counit"  )==0) { p = hdr.head2.counit;}
    else if (strcmp(key,"wevel"   )==0) { p = hdr.head2.wevel ;}
    else if (strcmp(key,"swevel"  )==0) { p = hdr.head2.swevel;}
    else if (strcmp(key,"sut"     )==0) { p = hdr.head2.sut   ;}
    else if (strcmp(key,"gut"     )==0) { p = hdr.head2.gut   ;}
    else if (strcmp(key,"sstat"   )==0) { p = hdr.head2.sstat ;}
    else if (strcmp(key,"gstat"   )==0) { p = hdr.head2.gstat ;}
    else if (strcmp(key,"tstat"   )==0) { p = hdr.head2.tstat ;}
    else if (strcmp(key,"laga"    )==0) { p = hdr.head2.laga  ;}
    else if (strcmp(key,"lagb"    )==0) { p = hdr.head2.lagb  ;}
    else if (strcmp(key,"delrt"   )==0) { p = hdr.head2.delrt ;}
    else if (strcmp(key,"muts"    )==0) { p = hdr.head2.muts  ;}
    else if (strcmp(key,"mute"    )==0) { p = hdr.head2.mute  ;}
    else if (strcmp(key,"ns"      )==0) { p = hdr.head2.ns    ;}
    else if (strcmp(key,"dt"      )==0) { p = hdr.head2.dt    ;}
    else if (strcmp(key,"gain"    )==0) { p = hdr.head2.gain  ;}
    else if (strcmp(key,"igc"     )==0) { p = hdr.head2.igc   ;}
    else if (strcmp(key,"igi"     )==0) { p = hdr.head2.igi   ;}
    else if (strcmp(key,"corr"    )==0) { p = hdr.head2.corr  ;}
    else if (strcmp(key,"sfs"     )==0) { p = hdr.head2.sfs   ;}
    else if (strcmp(key,"sfe"     )==0) { p = hdr.head2.sfe   ;}
    else if (strcmp(key,"slen"    )==0) { p = hdr.head2.slen  ;}
    else if (strcmp(key,"styp"    )==0) { p = hdr.head2.styp  ;}
    else if (strcmp(key,"stas"    )==0) { p = hdr.head2.stas  ;}
    else if (strcmp(key,"stae"    )==0) { p = hdr.head2.stae  ;}
    else if (strcmp(key,"tatyp"   )==0) { p = hdr.head2.tatyp ;}
    else if (strcmp(key,"afilf"   )==0) { p = hdr.head2.afilf ;}
    else if (strcmp(key,"afils"   )==0) { p = hdr.head2.afils ;}
    else if (strcmp(key,"nofilf"  )==0) { p = hdr.head2.nofilf;}
    else if (strcmp(key,"nofils"  )==0) { p = hdr.head2.nofils;}
    else if (strcmp(key,"lcf"     )==0) { p = hdr.head2.lcf   ;}
    else if (strcmp(key,"hcf"     )==0) { p = hdr.head2.hcf   ;}
    else if (strcmp(key,"lcs"     )==0) { p = hdr.head2.lcs   ;}
    else if (strcmp(key,"hcs"     )==0) { p = hdr.head2.hcs   ;}
    else if (strcmp(key,"year"    )==0) { p = hdr.head2.year  ;}
    else if (strcmp(key,"day"     )==0) { p = hdr.head2.day   ;}
    else if (strcmp(key,"hour"    )==0) { p = hdr.head2.hour  ;}
    else if (strcmp(key,"minute"  )==0) { p = hdr.head2.minute;}
    else if (strcmp(key,"sec"     )==0) { p = hdr.head2.sec   ;}
    else if (strcmp(key,"timbas"  )==0) { p = hdr.head2.timbas;}
    else if (strcmp(key,"trwf"    )==0) { p = hdr.head2.trwf  ;}
    else if (strcmp(key,"grnors"  )==0) { p = hdr.head2.grnors;}
    else if (strcmp(key,"grnofr"  )==0) { p = hdr.head2.grnofr;}
    else if (strcmp(key,"grnlof"  )==0) { p = hdr.head2.grnlof;}
    else if (strcmp(key,"gaps"    )==0) { p = hdr.head2.gaps  ;}
    else if (strcmp(key,"otrav"   )==0) { p = hdr.head2.otrav ;}
    else{p=0;std::cout<<"Error: Not find su key number!!";}
}
    return p;
}
float getSuHeadKey(segyhead2&hdr, const char *key, \
    bool doEndianSwap=false);
float getSuHeadKey(segyhead2&hdr, const char *key, bool doEndianSwap)
{
    float p;
if(!doEndianSwap){
    if(strcmp(key,"tracl"   )==0)       { p = (hdr.tracl);}
    else if (strcmp(key,"tracr"   )==0) { p = (hdr.tracr);}
    else if (strcmp(key,"fldr"    )==0) { p = (hdr.fldr) ;}
    else if (strcmp(key,"tracf"   )==0) { p = (hdr.tracf);}
    else if (strcmp(key,"ep"      )==0) { p = (hdr.ep)  ;}
    else if (strcmp(key,"cdp"     )==0) { p = (hdr.cdp) ;}
    else if (strcmp(key,"cdpt"    )==0) { p = (hdr.cdpt);}
    else if (strcmp(key,"trid"    )==0) { p = (hdr.trid);}
    else if (strcmp(key,"nvs"     )==0) { p = (hdr.nvs) ;}
    else if (strcmp(key,"nhs"     )==0) { p = hdr.nhs   ;}
    else if (strcmp(key,"duse"    )==0) { p = hdr.duse  ;}
    else if (strcmp(key,"offset"  )==0) { p = hdr.offset;}
    else if (strcmp(key,"gelev"   )==0) { p = hdr.gelev ;}
    else if (strcmp(key,"selev"   )==0) { p = hdr.selev ;}
    else if (strcmp(key,"sdepth"  )==0) { p = hdr.sdepth;}
    else if (strcmp(key,"gdel"    )==0) { p = hdr.gdel  ;}
    else if (strcmp(key,"sdel"    )==0) { p = hdr.sdel  ;}
    else if (strcmp(key,"swdep"   )==0) { p = hdr.swdep ;}
    else if (strcmp(key,"gwdep"   )==0) { p = hdr.gwdep ;}
    else if (strcmp(key,"scalel"  )==0) { p = hdr.scalel;}
    else if (strcmp(key,"scalco"  )==0) { p = hdr.scalco;}
    else if (strcmp(key,"sx"      )==0) { p = hdr.sx    ;}
    else if (strcmp(key,"sy"      )==0) { p = hdr.sy    ;}
    else if (strcmp(key,"gx"      )==0) { p = hdr.gx    ;}
    else if (strcmp(key,"gy"      )==0) { p = hdr.gy    ;}
    else if (strcmp(key,"counit"  )==0) { p = hdr.counit;}
    else if (strcmp(key,"wevel"   )==0) { p = hdr.wevel ;}
    else if (strcmp(key,"swevel"  )==0) { p = hdr.swevel;}
    else if (strcmp(key,"sut"     )==0) { p = hdr.sut   ;}
    else if (strcmp(key,"gut"     )==0) { p = hdr.gut   ;}
    else if (strcmp(key,"sstat"   )==0) { p = hdr.sstat ;}
    else if (strcmp(key,"gstat"   )==0) { p = hdr.gstat ;}
    else if (strcmp(key,"tstat"   )==0) { p = hdr.tstat ;}
    else if (strcmp(key,"laga"    )==0) { p = hdr.laga  ;}
    else if (strcmp(key,"lagb"    )==0) { p = hdr.lagb  ;}
    else if (strcmp(key,"delrt"   )==0) { p = hdr.delrt ;}
    else if (strcmp(key,"muts"    )==0) { p = hdr.muts  ;}
    else if (strcmp(key,"mute"    )==0) { p = hdr.mute  ;}
    else if (strcmp(key,"ns"      )==0) { p = hdr.ns    ;}
    else if (strcmp(key,"dt"      )==0) { p = hdr.dt    ;}
    else if (strcmp(key,"gain"    )==0) { p = hdr.gain  ;}
    else if (strcmp(key,"igc"     )==0) { p = hdr.igc   ;}
    else if (strcmp(key,"igi"     )==0) { p = hdr.igi   ;}
    else if (strcmp(key,"corr"    )==0) { p = hdr.corr  ;}
    else if (strcmp(key,"sfs"     )==0) { p = hdr.sfs   ;}
    else if (strcmp(key,"sfe"     )==0) { p = hdr.sfe   ;}
    else if (strcmp(key,"slen"    )==0) { p = hdr.slen  ;}
    else if (strcmp(key,"styp"    )==0) { p = hdr.styp  ;}
    else if (strcmp(key,"stas"    )==0) { p = hdr.stas  ;}
    else if (strcmp(key,"stae"    )==0) { p = hdr.stae  ;}
    else if (strcmp(key,"tatyp"   )==0) { p = hdr.tatyp ;}
    else if (strcmp(key,"afilf"   )==0) { p = hdr.afilf ;}
    else if (strcmp(key,"afils"   )==0) { p = hdr.afils ;}
    else if (strcmp(key,"nofilf"  )==0) { p = hdr.nofilf;}
    else if (strcmp(key,"nofils"  )==0) { p = hdr.nofils;}
    else if (strcmp(key,"lcf"     )==0) { p = hdr.lcf   ;}
    else if (strcmp(key,"hcf"     )==0) { p = hdr.hcf   ;}
    else if (strcmp(key,"lcs"     )==0) { p = hdr.lcs   ;}
    else if (strcmp(key,"hcs"     )==0) { p = hdr.hcs   ;}
    else if (strcmp(key,"year"    )==0) { p = hdr.year  ;}
    else if (strcmp(key,"day"     )==0) { p = hdr.day   ;}
    else if (strcmp(key,"hour"    )==0) { p = hdr.hour  ;}
    else if (strcmp(key,"minute"  )==0) { p = hdr.minute;}
    else if (strcmp(key,"sec"     )==0) { p = hdr.sec   ;}
    else if (strcmp(key,"timbas"  )==0) { p = hdr.timbas;}
    else if (strcmp(key,"trwf"    )==0) { p = hdr.trwf  ;}
    else if (strcmp(key,"grnors"  )==0) { p = hdr.grnors;}
    else if (strcmp(key,"grnofr"  )==0) { p = hdr.grnofr;}
    else if (strcmp(key,"grnlof"  )==0) { p = hdr.grnlof;}
    else if (strcmp(key,"gaps"    )==0) { p = hdr.gaps  ;}
    else if (strcmp(key,"otrav"   )==0) { p = hdr.otrav ;}
    else{p=0;std::cout<<"Error: Not find su key number!!";}
}else{
    if(strcmp(key,"tracl"   )==0)       { p = getendianchange(hdr.tracl);}
    else if (strcmp(key,"tracr"   )==0) { p = getendianchange(hdr.tracr);}
    else if (strcmp(key,"fldr"    )==0) { p = getendianchange(hdr.fldr) ;}
    else if (strcmp(key,"tracf"   )==0) { p = getendianchange(hdr.tracf);}
    else if (strcmp(key,"ep"      )==0) { p = getendianchange(hdr.ep)  ;}
    else if (strcmp(key,"cdp"     )==0) { p = getendianchange(hdr.cdp) ;}
    else if (strcmp(key,"cdpt"    )==0) { p = getendianchange(hdr.cdpt);}
    else if (strcmp(key,"trid"    )==0) { p = getendianchange(hdr.trid);}
    else if (strcmp(key,"nvs"     )==0) { p = getendianchange(hdr.nvs) ;}
    else if (strcmp(key,"nhs"     )==0) { p = getendianchange(hdr.nhs)   ;}
    else if (strcmp(key,"duse"    )==0) { p = getendianchange(hdr.duse)  ;}
    else if (strcmp(key,"offset"  )==0) { p = getendianchange(hdr.offset);}
    else if (strcmp(key,"gelev"   )==0) { p = getendianchange(hdr.gelev) ;}
    else if (strcmp(key,"selev"   )==0) { p = getendianchange(hdr.selev) ;}
    else if (strcmp(key,"sdepth"  )==0) { p = getendianchange(hdr.sdepth);}
    else if (strcmp(key,"gdel"    )==0) { p = getendianchange(hdr.gdel)  ;}
    else if (strcmp(key,"sdel"    )==0) { p = getendianchange(hdr.sdel)  ;}
    else if (strcmp(key,"swdep"   )==0) { p = getendianchange(hdr.swdep) ;}
    else if (strcmp(key,"gwdep"   )==0) { p = getendianchange(hdr.gwdep) ;}
    else if (strcmp(key,"scalel"  )==0) { p = getendianchange(hdr.scalel);}
    else if (strcmp(key,"scalco"  )==0) { p = getendianchange(hdr.scalco);}
    else if (strcmp(key,"sx"      )==0) { p = getendianchange(hdr.sx)    ;}
    else if (strcmp(key,"sy"      )==0) { p = getendianchange(hdr.sy)    ;}
    else if (strcmp(key,"gx"      )==0) { p = getendianchange(hdr.gx)    ;}
    else if (strcmp(key,"gy"      )==0) { p = getendianchange(hdr.gy)    ;}
    else if (strcmp(key,"counit"  )==0) { p = getendianchange(hdr.counit);}
    else if (strcmp(key,"wevel"   )==0) { p = getendianchange(hdr.wevel) ;}
    else if (strcmp(key,"swevel"  )==0) { p = getendianchange(hdr.swevel);}
    else if (strcmp(key,"sut"     )==0) { p = getendianchange(hdr.sut)   ;}
    else if (strcmp(key,"gut"     )==0) { p = getendianchange(hdr.gut)   ;}
    else if (strcmp(key,"sstat"   )==0) { p = getendianchange(hdr.sstat) ;}
    else if (strcmp(key,"gstat"   )==0) { p = getendianchange(hdr.gstat) ;}
    else if (strcmp(key,"tstat"   )==0) { p = getendianchange(hdr.tstat) ;}
    else if (strcmp(key,"laga"    )==0) { p = getendianchange(hdr.laga)  ;}
    else if (strcmp(key,"lagb"    )==0) { p = getendianchange(hdr.lagb)  ;}
    else if (strcmp(key,"delrt"   )==0) { p = getendianchange(hdr.delrt) ;}
    else if (strcmp(key,"muts"    )==0) { p = getendianchange(hdr.muts)  ;}
    else if (strcmp(key,"mute"    )==0) { p = getendianchange(hdr.mute)  ;}
    else if (strcmp(key,"ns"      )==0) { p = getendianchange(hdr.ns)    ;}
    else if (strcmp(key,"dt"      )==0) { p = getendianchange(hdr.dt)    ;}
    else if (strcmp(key,"gain"    )==0) { p = getendianchange(hdr.gain)  ;}
    else if (strcmp(key,"igc"     )==0) { p = getendianchange(hdr.igc)   ;}
    else if (strcmp(key,"igi"     )==0) { p = getendianchange(hdr.igi)   ;}
    else if (strcmp(key,"corr"    )==0) { p = getendianchange(hdr.corr)  ;}
    else if (strcmp(key,"sfs"     )==0) { p = getendianchange(hdr.sfs)   ;}
    else if (strcmp(key,"sfe"     )==0) { p = getendianchange(hdr.sfe)   ;}
    else if (strcmp(key,"slen"    )==0) { p = getendianchange(hdr.slen)  ;}
    else if (strcmp(key,"styp"    )==0) { p = getendianchange(hdr.styp)  ;}
    else if (strcmp(key,"stas"    )==0) { p = getendianchange(hdr.stas)  ;}
    else if (strcmp(key,"stae"    )==0) { p = getendianchange(hdr.stae)  ;}
    else if (strcmp(key,"tatyp"   )==0) { p = getendianchange(hdr.tatyp) ;}
    else if (strcmp(key,"afilf"   )==0) { p = getendianchange(hdr.afilf) ;}
    else if (strcmp(key,"afils"   )==0) { p = getendianchange(hdr.afils) ;}
    else if (strcmp(key,"nofilf"  )==0) { p = getendianchange(hdr.nofilf);}
    else if (strcmp(key,"nofils"  )==0) { p = getendianchange(hdr.nofils);}
    else if (strcmp(key,"lcf"     )==0) { p = getendianchange(hdr.lcf)   ;}
    else if (strcmp(key,"hcf"     )==0) { p = getendianchange(hdr.hcf)   ;}
    else if (strcmp(key,"lcs"     )==0) { p = getendianchange(hdr.lcs)   ;}
    else if (strcmp(key,"hcs"     )==0) { p = getendianchange(hdr.hcs)   ;}
    else if (strcmp(key,"year"    )==0) { p = getendianchange(hdr.year)  ;}
    else if (strcmp(key,"day"     )==0) { p = getendianchange(hdr.day)   ;}
    else if (strcmp(key,"hour"    )==0) { p = getendianchange(hdr.hour)  ;}
    else if (strcmp(key,"minute"  )==0) { p = getendianchange(hdr.minute);}
    else if (strcmp(key,"sec"     )==0) { p = getendianchange(hdr.sec)   ;}
    else if (strcmp(key,"timbas"  )==0) { p = getendianchange(hdr.timbas);}
    else if (strcmp(key,"trwf"    )==0) { p = getendianchange(hdr.trwf)  ;}
    else if (strcmp(key,"grnors"  )==0) { p = getendianchange(hdr.grnors);}
    else if (strcmp(key,"grnofr"  )==0) { p = getendianchange(hdr.grnofr);}
    else if (strcmp(key,"grnlof"  )==0) { p = getendianchange(hdr.grnlof);}
    else if (strcmp(key,"gaps"    )==0) { p = getendianchange(hdr.gaps)  ;}
    else if (strcmp(key,"otrav"   )==0) { p = getendianchange(hdr.otrav) ;}
    else{p=0;std::cout<<"Error: Not find su key number!!";}
}
    return p;
}

void* getPointSuHeadKey(segyhead2&hdr, const char *key, int& bitnum)
{
    void* p;
    if(strcmp(key,"tracl"   )==0)       
	{ p = &(hdr.tracl);bitnum=sizeof(hdr.tracl);}
    else if (strcmp(key,"tracr"   )==0) 
	{ p = &(hdr.tracr);bitnum=sizeof(hdr.tracr);}
    else if (strcmp(key,"fldr"    )==0) 
	{ p = &(hdr.fldr) ;bitnum=sizeof(hdr.fldr);}
    else if (strcmp(key,"tracf"   )==0) 
	{ p = &(hdr.tracf);bitnum=sizeof(hdr.tracf);}
    else if (strcmp(key,"ep"      )==0) 
	{ p = &(hdr.ep)  ;bitnum=sizeof(hdr.ep);}
    else if (strcmp(key,"cdp"     )==0) 
	{ p = &(hdr.cdp) ;bitnum=sizeof(hdr.cdp);}
    else if (strcmp(key,"cdpt"    )==0) 
	{ p = &(hdr.cdpt);bitnum=sizeof(hdr.cdpt);}
    else if (strcmp(key,"trid"    )==0) 
	{ p = &(hdr.trid);bitnum=sizeof(hdr.trid);}
    else if (strcmp(key,"nvs"     )==0) 
	{ p = &(hdr.nvs) ;bitnum=sizeof(hdr.nvs);}
    else if (strcmp(key,"nhs"     )==0) 
	{ p = &hdr.nhs   ;bitnum=sizeof(hdr.nhs);}
    else if (strcmp(key,"duse"    )==0) 
	{ p = &hdr.duse  ;bitnum=sizeof(hdr.duse);}
    else if (strcmp(key,"offset"  )==0) 
	{ p = &hdr.offset;bitnum=sizeof(hdr.offset);}
    else if (strcmp(key,"gelev"   )==0) 
	{ p = &hdr.gelev ;bitnum=sizeof(hdr.gelev);}
    else if (strcmp(key,"selev"   )==0) 
	{ p = &hdr.selev ;bitnum=sizeof(hdr.selev);}
    else if (strcmp(key,"sdepth"  )==0) 
	{ p = &hdr.sdepth;bitnum=sizeof(hdr.sdepth);}
    else if (strcmp(key,"gdel"    )==0) 
	{ p = &hdr.gdel  ;bitnum=sizeof(hdr.gdel);}
    else if (strcmp(key,"sdel"    )==0) 
	{ p = &hdr.sdel  ;bitnum=sizeof(hdr.sdel);}
    else if (strcmp(key,"swdep"   )==0) 
	{ p = &hdr.swdep ;bitnum=sizeof(hdr.swdep);}
    else if (strcmp(key,"gwdep"   )==0) 
	{ p = &hdr.gwdep ;bitnum=sizeof(hdr.gwdep);}
    else if (strcmp(key,"scalel"  )==0) 
	{ p = &hdr.scalel;bitnum=sizeof(hdr.scalel);}
    else if (strcmp(key,"scalco"  )==0) 
	{ p = &hdr.scalco;bitnum=sizeof(hdr.scalco);}
    else if (strcmp(key,"sx"      )==0) 
	{ p = &hdr.sx    ;bitnum=sizeof(hdr.sx);}
    else if (strcmp(key,"sy"      )==0) 
	{ p = &hdr.sy    ;bitnum=sizeof(hdr.sy);}
    else if (strcmp(key,"gx"      )==0) 
	{ p = &hdr.gx    ;bitnum=sizeof(hdr.gx);}
    else if (strcmp(key,"gy"      )==0) 
	{ p = &hdr.gy    ;bitnum=sizeof(hdr.gy);}
    else if (strcmp(key,"counit"  )==0) 
	{ p = &hdr.counit;bitnum=sizeof(hdr.counit);}
    else if (strcmp(key,"wevel"   )==0) 
	{ p = &hdr.wevel ;bitnum=sizeof(hdr.wevel);}
    else if (strcmp(key,"swevel"  )==0) 
	{ p = &hdr.swevel;bitnum=sizeof(hdr.swevel);}
    else if (strcmp(key,"sut"     )==0) 
	{ p = &hdr.sut   ;bitnum=sizeof(hdr.sut);}
    else if (strcmp(key,"gut"     )==0) 
	{ p = &hdr.gut   ;bitnum=sizeof(hdr.gut);}
    else if (strcmp(key,"sstat"   )==0) 
	{ p = &hdr.sstat ;bitnum=sizeof(hdr.sstat);}
    else if (strcmp(key,"gstat"   )==0) 
	{ p = &hdr.gstat ;bitnum=sizeof(hdr.gstat);}
    else if (strcmp(key,"tstat"   )==0) 
	{ p = &hdr.tstat ;bitnum=sizeof(hdr.tstat);}
    else if (strcmp(key,"laga"    )==0) 
	{ p = &hdr.laga  ;bitnum=sizeof(hdr.laga);}
    else if (strcmp(key,"lagb"    )==0) 
	{ p = &hdr.lagb  ;bitnum=sizeof(hdr.lagb);}
    else if (strcmp(key,"delrt"   )==0) 
	{ p = &hdr.delrt ;bitnum=sizeof(hdr.delrt);}
    else if (strcmp(key,"muts"    )==0) 
	{ p = &hdr.muts  ;bitnum=sizeof(hdr.muts);}
    else if (strcmp(key,"mute"    )==0) 
	{ p = &hdr.mute  ;bitnum=sizeof(hdr.mute);}
    else if (strcmp(key,"ns"      )==0) 
	{ p = &hdr.ns    ;bitnum=sizeof(hdr.ns);}
    else if (strcmp(key,"dt"      )==0) 
	{ p = &hdr.dt    ;bitnum=sizeof(hdr.dt);}
    else if (strcmp(key,"gain"    )==0) 
	{ p = &hdr.gain  ;bitnum=sizeof(hdr.gain);}
    else if (strcmp(key,"igc"     )==0) 
	{ p = &hdr.igc   ;bitnum=sizeof(hdr.igc);}
    else if (strcmp(key,"igi"     )==0) 
	{ p = &hdr.igi   ;bitnum=sizeof(hdr.igi);}
    else if (strcmp(key,"corr"    )==0) 
	{ p = &hdr.corr  ;bitnum=sizeof(hdr.corr);}
    else if (strcmp(key,"sfs"     )==0) 
	{ p = &hdr.sfs   ;bitnum=sizeof(hdr.sfs);}
    else if (strcmp(key,"sfe"     )==0) 
	{ p = &hdr.sfe   ;bitnum=sizeof(hdr.sfe);}
    else if (strcmp(key,"slen"    )==0) 
	{ p = &hdr.slen  ;bitnum=sizeof(hdr.slen);}
    else if (strcmp(key,"styp"    )==0) 
	{ p = &hdr.styp  ;bitnum=sizeof(hdr.styp);}
    else if (strcmp(key,"stas"    )==0) 
	{ p = &hdr.stas  ;bitnum=sizeof(hdr.stas);}
    else if (strcmp(key,"stae"    )==0) 
	{ p = &hdr.stae  ;bitnum=sizeof(hdr.stae);}
    else if (strcmp(key,"tatyp"   )==0) 
	{ p = &hdr.tatyp ;bitnum=sizeof(hdr.tatyp);}
    else if (strcmp(key,"afilf"   )==0) 
	{ p = &hdr.afilf ;bitnum=sizeof(hdr.afilf);}
    else if (strcmp(key,"afils"   )==0) 
	{ p = &hdr.afils ;bitnum=sizeof(hdr.afils);}
    else if (strcmp(key,"nofilf"  )==0) 
	{ p = &hdr.nofilf;bitnum=sizeof(hdr.nofilf);}
    else if (strcmp(key,"nofils"  )==0) 
	{ p = &hdr.nofils;bitnum=sizeof(hdr.nofils);}
    else if (strcmp(key,"lcf"     )==0) 
	{ p = &hdr.lcf   ;bitnum=sizeof(hdr.lcf);}
    else if (strcmp(key,"hcf"     )==0) 
	{ p = &hdr.hcf   ;bitnum=sizeof(hdr.hcf);}
    else if (strcmp(key,"lcs"     )==0) 
	{ p = &hdr.lcs   ;bitnum=sizeof(hdr.lcs);}
    else if (strcmp(key,"hcs"     )==0) 
	{ p = &hdr.hcs   ;bitnum=sizeof(hdr.hcs);}
    else if (strcmp(key,"year"    )==0) 
	{ p = &hdr.year  ;bitnum=sizeof(hdr.year);}
    else if (strcmp(key,"day"     )==0) 
	{ p = &hdr.day   ;bitnum=sizeof(hdr.day);}
    else if (strcmp(key,"hour"    )==0) 
	{ p = &hdr.hour  ;bitnum=sizeof(hdr.hour);}
    else if (strcmp(key,"minute"  )==0) 
	{ p = &hdr.minute;bitnum=sizeof(hdr.minute);}
    else if (strcmp(key,"sec"     )==0) 
	{ p = &hdr.sec   ;bitnum=sizeof(hdr.sec);}
    else if (strcmp(key,"timbas"  )==0) 
	{ p = &hdr.timbas;bitnum=sizeof(hdr.timbas);}
    else if (strcmp(key,"trwf"    )==0) 
	{ p = &hdr.trwf  ;bitnum=sizeof(hdr.trwf);}
    else if (strcmp(key,"grnors"  )==0) 
	{ p = &hdr.grnors;bitnum=sizeof(hdr.grnors);}
    else if (strcmp(key,"grnofr"  )==0) 
	{ p = &hdr.grnofr;bitnum=sizeof(hdr.grnofr);}
    else if (strcmp(key,"grnlof"  )==0) 
	{ p = &hdr.grnlof;bitnum=sizeof(hdr.grnlof);}
    else if (strcmp(key,"gaps"    )==0) 
	{ p = &hdr.gaps  ;bitnum=sizeof(hdr.gaps);}
    else if (strcmp(key,"otrav"   )==0) 
	{ p = &hdr.otrav ;bitnum=sizeof(hdr.otrav);}
    else{p=0;std::cout<<"Error: Not find su key number!!";}
    return p;
}
/***********************************************************************
ibm_to_float - convert between 32 bit IBM and IEEE floating numbers
************************************************************************
Input::
from		input vector
to		output vector, can be same as input vector
endian		byte order =0 little endian (DEC, PC's)
			    =1 other systems
*************************************************************************
Notes:
Up to 3 bits lost on IEEE -> IBM

Assumes sizeof(int) == 4

IBM -> IEEE may overflow or underflow, taken care of by
substituting large number or zero

Only integer shifting and masking are used.
*************************************************************************
Credits: CWP: Brian Sumner,  c.1985
*************************************************************************/
float ibm2ieee (float fromf) 
{
    float to;
    int *from = (int *)&fromf;
    // int fconv, fmant, i, t;
    int fconv, fmant, t;
    fconv = from[0];

    if (fconv)
    {
        fmant = 0x00ffffff & fconv;
        t = (int)((0x7f000000 & fconv) >> 22) - 130;
        while (!(fmant & 0x00800000))
        {
            --t;
            fmant <<= 1;
        }
        if (t > 254)
            fconv = (0x80000000 & fconv) | 0x7f7fffff;
        else if (t <= 0)
            fconv = 0;
        else
            fconv = (0x80000000 & fconv) | (t << 23) | (0x007fffff & fmant);
    }
    to = fconv;
    return to;
}
/*
unsigned int ibm2ieee (const unsigned int num) 
{  
    
    unsigned int ibmCode(num);
    //unsigned int ieeeFloat(0);
    //unsigned int ieeeFloat(0);
if((ibmCode << 1) == 0)
{  
// If it is an IBM floating point number  
// Returns the corresponding IEEE binary encoding  
    return ibmCode;
}
//IBM浮点数： SEEEEEEE MMMMMMMM MMMMMMMM MMMMMMMM
//浮点数值 Value = (-1)^s * M * 16^(E-64)
   unsigned int  signCode(num>>31);//获取符号位, S=0或1， sign=00 00 00 0000000S

   ibmCode= (ibmCode<<1); // 左移移出符号位, ibmCode= EEEEEEEM MMMMMMMM MMMMMMMM MMMMMMM0

    unsigned int float_IBM(num);

// Get the S, symbol part of the IBM floating point number  
unsigned int S_IBM_32(signCode<<31);  
// Get the E, exponential part of the IBM floating point number  
unsigned int exponent(ibmCode >> 25);
//int exponent2((int)(ibmCode >> 25));
unsigned int E_IBM_32(exponent << 24);  
// Get the F, decimal part of the IBM floating point number 
unsigned int fraction(ibmCode << 7);  //移出符号位和阶数剩余的部分：尾数部分fraction=MMMMMMMM MMMMMMMM MMMMMMM 00000000
   fraction= (fraction >> 8);  //fraction=00000000 MMMMMMMM MMMMMMMM MMMMMMM
unsigned int F_IBM_32(fraction);  
// Get the F, decimal part of the IBM floating point number  
unsigned int radix(0);  
unsigned int F_IEEE_32(F_IBM_32);  
while (radix <= 3 && F_IEEE_32 < 0x01000000) {  
    radix++;  
    F_IEEE_32 = F_IEEE_32 << 1;  }  
// Put it back in the appropriate position in the IEEE type F section  
F_IEEE_32 = (F_IEEE_32 - 0x01000000)>>1;  
// Get the E, exponential part of the IBM floating point number  
 // Start counting  
// Put it in the correct position  

unsigned int E_IEEE_32((((E_IBM_32>>22)-130)-(radix-1))<<23);  
// Whether overflow occurs  
if (E_IEEE_32 > 0x7F000000) {  
    //return S_IBM_32|0x7F000000;  }  
    return S_IBM_32;  }  
if (E_IEEE_32 < 0x10000000) {  
    return S_IBM_32;  } 
else  
    //return ieeeFloat;  
    return S_IBM_32 | E_IEEE_32 | F_IEEE_32;  
}  
*/
////////////////////////////////////////////////
void Bubble_sort(float *data, int n)
{
    int i, j;
    float t;
    for (j = 1; j <= n; j++)
    {
        for (i = 0; i <= n - j; i++)
        {
            if (data[i] > data[i + 1])
            {
                t = data[i];
                data[i] = data[i + 1];
                data[i + 1] = t;
            }
        }
    }
}

void swap_float_4(float *tnf4)
{
    int *tni4 = (int *)tnf4;
    *tni4 = (((*tni4 >> 24) & 0xff) | ((*tni4 & 0xff) << 24) |
             ((*tni4 >> 8) & 0xff00) | ((*tni4 & 0xff00) << 8));
}

float efloat(float value)
{
    union
    {
        char bytes[4];
        float flt;
    } ieee;

    // swap bytes
    unsigned char temp;
    ieee.flt = value;

    temp = ieee.bytes[3];
    ieee.bytes[3] = ieee.bytes[0];
    ieee.bytes[0] = temp;

    temp = ieee.bytes[2];
    ieee.bytes[2] = ieee.bytes[1];
    ieee.bytes[1] = temp;

    return (ieee.flt);
}

void ibm_to_float(unsigned int from[], unsigned int to[], int n, int endian)
/***********************************************************************
ibm_to_float - convert between 32 bit IBM and IEEE floating numbers
************************************************************************
Input::
from		input vector
to		output vector, can be same as input vector
endian		byte order =0 little endian (DEC, PC's)
			    =1 other systems
*************************************************************************
Notes:
Up to 3 bits lost on IEEE -> IBM

Assumes sizeof(int) == 4

IBM -> IEEE may overflow or underflow, taken care of by
substituting large number or zero

Only integer shifting and masking are used.
*************************************************************************
Credits: CWP: Brian Sumner,  c.1985
*************************************************************************/
{
    unsigned int fconv, fmant, i, t;

    for (i = 0; i < n; ++i)
    {
        fconv = from[i];
        /* if little endian, i.e. endian=0 do this */
        if (endian == 0)
            fconv = (fconv << 24) | ((fconv >> 24) & 0xff) |
                    ((fconv & 0xff00) << 8) | ((fconv & 0xff0000) >> 8);
        if (fconv)
        {
            fmant = 0x00ffffff & fconv;
            /* The next two lines were added by Toralf Foerster */
            /* to trap non-IBM format data i.e. conv=0 data  */
            if (fmant == 0)
                printf("mantissa is zero data may not be in IBM FLOAT Format !");
            t = (unsigned int)((0x7f000000 & fconv) >> 22) - 130;
            while (!(fmant & 0x00800000))
            {
                --t;
                fmant <<= 1;
            }
            if (t > 254)
                fconv = (0x80000000 & fconv) | 0x7f7fffff;
            else if (t <= 0)
                fconv = 0;
            else
                fconv = (0x80000000 & fconv) | (t << 23) | (0x007fffff & fmant);
        }
        to[i] = fconv;
    }
    return;
}

void float_to_ibm(int from[], int to[], int n, int endian)
/**********************************************************************
 float_to_ibm - convert between 32 bit IBM and IEEE floating numbers
***********************************************************************
Input:
from	   input vector
n	   number of floats in vectors
endian	   =0 for little endian machine, =1 for big endian machines

Output:
to	   output vector, can be same as input vector

***********************************************************************
Notes:
Up to 3 bits lost on IEEE -> IBM

IBM -> IEEE may overflow or underflow, taken care of by
substituting large number or zero

Only integer shifting and masking are used.
***********************************************************************
Credits:     CWP: Brian Sumner
***********************************************************************/
{
    register int fconv, fmant, i, t;

    for (i = 0; i < n; ++i)
    {
        fconv = from[i];
        if (fconv)
        {
            fmant = (0x007fffff & fconv) | 0x00800000;
            t = (int)((0x7f800000 & fconv) >> 23) - 126;
            while (t & 0x3)
            {
                ++t;
                fmant >>= 1;
            }
            fconv = (0x80000000 & fconv) | (((t >> 2) + 64) << 24) | fmant;
        }
        if (endian == 0)
            fconv = (fconv << 24) | ((fconv >> 24) & 0xff) |
                    ((fconv & 0xff00) << 8) | ((fconv & 0xff0000) >> 8);

        to[i] = fconv;
    }
    return;
}

#endif
