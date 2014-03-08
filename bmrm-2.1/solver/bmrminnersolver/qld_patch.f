      SUBROUTINE QL0001(M,ME,MMAX,N,NMAX,MNN,C,D,A,B,XL,XU,
     1           X,U,IOUT,IFAIL,IPRINT,WAR,LWAR,IWAR,LIWAR)
c
cCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
c
c                     !!!! NOTICE !!!!
c
c 1. The routines contained in this file are due to Prof. K.Schittkowski
c    of the University of Bayreuth, Germany (modification of routines
c    due to Prof. MJD Powell at the University of Cambridge).  They can
c    be freely distributed.
c
c 2. A minor modification was performed at the University of Maryland. 
c    It is marked in the code by "c umd".
c
c                                      A.L. Tits and J.L. Zhou
c                                      University of Maryland
C
C***********************************************************************
C
C
C             SOLUTION OF QUADRATIC PROGRAMMING PROBLEMS
C
C
C
C   QL0001 SOLVES THE QUADRATIC PROGRAMMING PROBLEM
C
C   MINIMIZE        .5*X'*C*X + D'*X
C   SUBJECT TO      A(J)*X  +  B(J)   =  0  ,  J=1,...,ME
C                   A(J)*X  +  B(J)  >=  0  ,  J=ME+1,...,M
C                   XL  <=  X  <=  XU
C   
C   HERE C MUST BE AN N BY N SYMMETRIC AND POSITIVE MATRIX, D AN N-DIMENSIONAL
C   VECTOR, A AN M BY N MATRIX AND B AN M-DIMENSIONAL VECTOR. THE ABOVE 
C   SITUATION IS INDICATED BY IWAR(1)=1. ALTERNATIVELY, I.E. IF IWAR(1)=0,
C   THE OBJECTIVE FUNCTION MATRIX CAN ALSO BE PROVIDED IN FACTORIZED FORM.
C   IN THIS CASE, C IS AN UPPER TRIANGULAR MATRIX.
C
C   THE SUBROUTINE REORGANIZES SOME DATA SO THAT THE PROBLEM CAN BE SOLVED
C   BY A MODIFICATION OF AN ALGORITHM PROPOSED BY POWELL (1983).
C
C
C   USAGE:
C
C      QL0001(M,ME,MMAX,N,NMAX,MNN,C,D,A,B,XL,XU,X,U,IOUT,IFAIL,IPRINT,
C             WAR,LWAR,IWAR,LIWAR)
C
C
C   DEFINITION OF THE PARAMETERS:
C
C   M :        TOTAL NUMBER OF CONSTRAINTS.
C   ME :       NUMBER OF EQUALITY CONSTRAINTS.
C   MMAX :     ROW DIMENSION OF A. MMAX MUST BE AT LEAST ONE AND GREATER
C              THAN M.
C   N :        NUMBER OF VARIABLES.
C   NMAX :     ROW DIMENSION OF C. NMAX MUST BE GREATER OR EQUAL TO N.
C   MNN :      MUST BE EQUAL TO M + N + N.
C   C(NMAX,NMAX): OBJECTIVE FUNCTION MATRIX WHICH SHOULD BE SYMMETRIC AND
C              POSITIVE DEFINITE. IF IWAR(1) = 0, C IS SUPPOSED TO BE THE
C              CHOLESKEY-FACTOR OF ANOTHER MATRIX, I.E. C IS UPPER
C              TRIANGULAR.
C   D(NMAX) :  CONTAINS THE CONSTANT VECTOR OF THE OBJECTIVE FUNCTION.
C   A(MMAX,NMAX): CONTAINS THE DATA MATRIX OF THE LINEAR CONSTRAINTS.
C   B(MMAX) :  CONTAINS THE CONSTANT DATA OF THE LINEAR CONSTRAINTS.
C   XL(N),XU(N): CONTAIN THE LOWER AND UPPER BOUNDS FOR THE VARIABLES.
C   X(N) :     ON RETURN, X CONTAINS THE OPTIMAL SOLUTION VECTOR.
C   U(MNN) :   ON RETURN, U CONTAINS THE LAGRANGE MULTIPLIERS. THE FIRST
C              M POSITIONS ARE RESERVED FOR THE MULTIPLIERS OF THE M
C              LINEAR CONSTRAINTS AND THE SUBSEQUENT ONES FOR THE 
C              MULTIPLIERS OF THE LOWER AND UPPER BOUNDS. ON SUCCESSFUL
C              TERMINATION, ALL VALUES OF U WITH RESPECT TO INEQUALITIES 
C              AND BOUNDS SHOULD BE GREATER OR EQUAL TO ZERO.
C   IOUT :     INTEGER INDICATING THE DESIRED OUTPUT UNIT NUMBER, I.E.
C              ALL WRITE-STATEMENTS START WITH 'WRITE(IOUT,... '.
C   IFAIL :    SHOWS THE TERMINATION REASON.
C      IFAIL = 0 :   SUCCESSFUL RETURN.
C      IFAIL = 1 :   TOO MANY ITERATIONS (MORE THAN 40*(N+M)).
C      IFAIL = 2 :   ACCURACY INSUFFICIENT TO SATISFY CONVERGENCE
C                    CRITERION.
C      IFAIL = 5 :   LENGTH OF A WORKING ARRAY IS TOO SHORT.
C      IFAIL > 10 :  THE CONSTRAINTS ARE INCONSISTENT.
C   IPRINT :   OUTPUT CONTROL.
C      IPRINT = 0 :  NO OUTPUT OF QL0001.
C      IPRINT > 0 :  BRIEF OUTPUT IN ERROR CASES.
C   WAR(LWAR) : REAL WORKING ARRAY. THE LENGTH LWAR SHOULD BE GRATER THAN
C               3*NMAX*NMAX/2 + 10*NMAX + 2*MMAX.
C   IWAR(LIWAR): INTEGER WORKING ARRAY. THE LENGTH LIWAR SHOULD BE AT
C              LEAST N.
C              IF IWAR(1)=0 INITIALLY, THEN THE CHOLESKY DECOMPOSITION
C              WHICH IS REQUIRED BY THE DUAL ALGORITHM TO GET THE FIRST
C              UNCONSTRAINED MINIMUM OF THE OBJECTIVE FUNCTION, IS
C              PERFORMED INTERNALLY. OTHERWISE, I.E. IF IWAR(1)=1, THEN
C              IT IS ASSUMED THAT THE USER PROVIDES THE INITIAL FAC-
C              TORIZATION BY HIMSELF AND STORES IT IN THE UPPER TRIAN-
C              GULAR PART OF THE ARRAY C.
C
C   A NAMED COMMON-BLOCK  /CMACHE/EPS   MUST BE PROVIDED BY THE USER,
C   WHERE EPS DEFINES A GUESS FOR THE UNDERLYING MACHINE PRECISION.
C
C
C   AUTHOR:    K. SCHITTKOWSKI,
C              MATHEMATISCHES INSTITUT,
C              UNIVERSITAET BAYREUTH,
C              8580 BAYREUTH,
C              GERMANY, F.R.
C
C
C   VERSION:   1.4  (MARCH, 1987)  
C
C
C*********************************************************************
C
C
      INTEGER NMAX,MMAX,N,MNN,LWAR,LIWAR
      DIMENSION C(NMAX,NMAX),D(NMAX),A(MMAX,NMAX),B(MMAX),
     1      XL(N),XU(N),X(N),U(MNN),WAR(LWAR),IWAR(LIWAR)
      DOUBLE PRECISION C,D,A,B,X,XL,XU,U,WAR,DIAG,ZERO,
     1      EPS,QPEPS,TEN
      INTEGER M,ME,IOUT,IFAIL,IPRINT,IWAR,INW1,INW2,IN,J,LW,MN,I,         
     1      IDIAG,INFO,NACT,MAXIT
      LOGICAL LQL
C
C     INTRINSIC FUNCTIONS:  DSQRT
C
      EPS=2.23e-16
C
C     CONSTANT DATA
C
c#################################################################
c

      if(dabs(c(nmax,nmax)).lt.eps) c(nmax,nmax)=eps
c
c umd
c  This prevents a subsequent more major modification of the Hessian
c  matrix in the important case when a minmax problem (yielding a 
c  singular Hessian matrix) is being solved.
c                                 ----UMCP, April 1991, Jian L. Zhou
c#################################################################
c
      LQL=.FALSE.
      IF (IWAR(1).EQ.1) LQL=.TRUE.
      ZERO=0.0D+0
      TEN=1.D+1
      MAXIT=40*(M+N)
      QPEPS=EPS
      INW1=1
      INW2=INW1+MMAX
C
C     PREPARE PROBLEM DATA FOR EXECUTION
C
      IF (M.LE.0) GOTO 20
      IN=INW1
      DO 10 J=1,M
      WAR(IN)=-B(J)
   10 IN=IN+1
   20 LW=3*NMAX*NMAX/2 + 10*NMAX + M
      IF ((INW2+LW).GT.LWAR) GOTO 80
      IF (LIWAR.LT.N) GOTO 81
      IF (MNN.LT.M+N+N) GOTO 82
      MN=M+N
C
C     CALL OF QL0002
C
      CALL QL0002(N,M,ME,MMAX,MN,MNN,NMAX,LQL,A,WAR(INW1),
     1    D,C,XL,XU,X,NACT,IWAR,MAXIT,QPEPS,INFO,DIAG,
     2    WAR(INW2),LW)
C
C     TEST OF MATRIX CORRECTIONS
C
      IFAIL=0
      IF (INFO.EQ.1) GOTO 40
      IF (INFO.EQ.2) GOTO 90
      IDIAG=0
      IF ((DIAG.GT.ZERO).AND.(DIAG.LT.1000.0)) IDIAG=DIAG
      IF ((IPRINT.GT.0).AND.(IDIAG.GT.0))
     1   WRITE(IOUT,1000) IDIAG
      IF (INFO .LT. 0) GOTO  70
C
C     REORDER MULTIPLIER
C
      DO  50 J=1,MNN
   50 U(J)=ZERO
      IN=INW2-1
      IF (NACT.EQ.0) GOTO 30
      DO  60 I=1,NACT
      J=IWAR(I)
      U(J)=WAR(IN+I)
   60 CONTINUE
   30 CONTINUE
      RETURN
C
C     ERROR MESSAGES
C
   70 IFAIL=-INFO+10
      IF ((IPRINT.GT.0).AND.(NACT.GT.0))
     1     WRITE(IOUT,1100) -INFO,(IWAR(I),I=1,NACT)
      RETURN
   80 IFAIL=5
      IF (IPRINT .GT. 0) WRITE(IOUT,1200)
      RETURN
   81 IFAIL=5
      IF (IPRINT .GT. 0) WRITE(IOUT,1210)
      RETURN
   82 IFAIL=5
      IF (IPRINT .GT. 0) WRITE(IOUT,1220)
      RETURN
   40 IFAIL=1
      IF (IPRINT.GT.0) WRITE(IOUT,1300) MAXIT
      RETURN
   90 IFAIL=2
      IF (IPRINT.GT.0) WRITE(IOUT,1400) 
      RETURN
C
C     FORMAT-INSTRUCTIONS
C
 1000 FORMAT(/8X,28H***QL: MATRIX G WAS ENLARGED,I3,
     1        20H-TIMES BY UNITMATRIX)
 1100 FORMAT(/8X,18H***QL: CONSTRAINT ,I5,
     1        19H NOT CONSISTENT TO ,/,(10X,10I5))
 1200 FORMAT(/8X,21H***QL: LWAR TOO SMALL)
 1210 FORMAT(/8X,22H***QL: LIWAR TOO SMALL)
 1220 FORMAT(/8X,20H***QL: MNN TOO SMALL)
 1300 FORMAT(/8X,37H***QL: TOO MANY ITERATIONS (MORE THAN,I6,1H))
 1400 FORMAT(/8X,50H***QL: ACCURACY INSUFFICIENT TO ATTAIN CONVERGENCE) 
      END
C
      SUBROUTINE QL0002(N,M,MEQ,MMAX,MN,MNN,NMAX,LQL,A,B,GRAD,G,
     1      XL,XU,X,NACT,IACT,MAXIT,VSMALL,INFO,DIAG,W,LW)
C
C**************************************************************************
C
C
C   THIS SUBROUTINE SOLVES THE QUADRATIC PROGRAMMING PROBLEM 
C
C       MINIMIZE      GRAD'*X  +  0.5 * X*G*X
C       SUBJECT TO    A(K)*X  =  B(K)   K=1,2,...,MEQ,
C                     A(K)*X >=  B(K)   K=MEQ+1,...,M,
C                     XL  <=  X  <=  XU
C
C   THE QUADRATIC PROGRAMMING METHOD PROCEEDS FROM AN INITIAL CHOLESKY-
C   DECOMPOSITION OF THE OBJECTIVE FUNCTION MATRIX, TO CALCULATE THE
C   UNIQUELY DETERMINED MINIMIZER OF THE UNCONSTRAINED PROBLEM. 
C   SUCCESSIVELY ALL VIOLATED CONSTRAINTS ARE ADDED TO A WORKING SET 
C   AND A MINIMIZER OF THE OBJECTIVE FUNCTION SUBJECT TO ALL CONSTRAINTS 
C   IN THIS WORKING SET IS COMPUTED. IT IS POSSIBLE THAT CONSTRAINTS
C   HAVE TO LEAVE THE WORKING SET.
C
C
C   DESCRIPTION OF PARAMETERS:    
C
C     N        : IS THE NUMBER OF VARIABLES.
C     M        : TOTAL NUMBER OF CONSTRAINTS.
C     MEQ      : NUMBER OF EQUALITY CONTRAINTS.
C     MMAX     : ROW DIMENSION OF A, DIMENSION OF B. MMAX MUST BE AT
C                LEAST ONE AND GREATER OR EQUAL TO M.
C     MN       : MUST BE EQUAL M + N.
C     MNN      : MUST BE EQUAL M + N + N.
C     NMAX     : ROW DIEMSION OF G. MUST BE AT LEAST N.
C     LQL      : DETERMINES INITIAL DECOMPOSITION.
C        LQL = .FALSE.  : THE UPPER TRIANGULAR PART OF THE MATRIX G
C                         CONTAINS INITIALLY THE CHOLESKY-FACTOR OF A SUITABLE 
C                         DECOMPOSITION.
C        LQL = .TRUE.   : THE INITIAL CHOLESKY-FACTORISATION OF G IS TO BE
C                         PERFORMED BY THE ALGORITHM.
C     A(MMAX,NMAX) : A IS A MATRIX WHOSE COLUMNS ARE THE CONSTRAINTS NORMALS.
C     B(MMAX)  : CONTAINS THE RIGHT HAND SIDES OF THE CONSTRAINTS.
C     GRAD(N)  : CONTAINS THE OBJECTIVE FUNCTION VECTOR GRAD.
C     G(NMAX,N): CONTAINS THE SYMMETRIC OBJECTIVE FUNCTION MATRIX.
C     XL(N), XU(N): CONTAIN THE LOWER AND UPPER BOUNDS FOR X.
C     X(N)     : VECTOR OF VARIABLES.
C     NACT     : FINAL NUMBER OF ACTIVE CONSTRAINTS.
C     IACT(K) (K=1,2,...,NACT): INDICES OF THE FINAL ACTIVE CONSTRAINTS.
C     INFO     : REASON FOR THE RETURN FROM THE SUBROUTINE.
C         INFO = 0 : CALCULATION WAS TERMINATED SUCCESSFULLY.
C         INFO = 1 : MAXIMUM NUMBER OF ITERATIONS ATTAINED.
C         INFO = 2 : ACCURACY IS INSUFFICIENT TO MAINTAIN INCREASING
C                    FUNCTION VALUES.
C         INFO < 0 : THE CONSTRAINT WITH INDEX ABS(INFO) AND THE CON-
C                    STRAINTS WHOSE INDICES ARE IACT(K), K=1,2,...,NACT,
C                    ARE INCONSISTENT.
C     MAXIT    : MAXIMUM NUMBER OF ITERATIONS.
C     VSMALL   : REQUIRED ACCURACY TO BE ACHIEVED (E.G. IN THE ORDER OF THE 
C                MACHINE PRECISION FOR SMALL AND WELL-CONDITIONED PROBLEMS).
C     DIAG     : ON RETURN DIAG IS EQUAL TO THE MULTIPLE OF THE UNIT MATRIX
C                THAT WAS ADDED TO G TO ACHIEVE POSITIVE DEFINITENESS.
C     W(LW)    : THE ELEMENTS OF W(.) ARE USED FOR WORKING SPACE. THE LENGTH
C                OF W MUST NOT BE LESS THAN (1.5*NMAX*NMAX + 10*NMAX + M).
C                WHEN INFO = 0 ON RETURN, THE LAGRANGE MULTIPLIERS OF THE
C                FINAL ACTIVE CONSTRAINTS ARE HELD IN W(K), K=1,2,...,NACT.
C   THE VALUES OF N, M, MEQ, MMAX, MN, MNN AND NMAX AND THE ELEMENTS OF
C   A, B, GRAD AND G ARE NOT ALTERED.
C
C   THE FOLLOWING INTEGERS ARE USED TO PARTITION W:
C     THE FIRST N ELEMENTS OF W HOLD LAGRANGE MULTIPLIER ESTIMATES.
C     W(IWZ+I+(N-1)*J) HOLDS THE MATRIX ELEMENT Z(I,J).
C     W(IWR+I+0.5*J*(J-1)) HOLDS THE UPPER TRIANGULAR MATRIX
C       ELEMENT R(I,J). THE SUBSEQUENT N COMPONENTS OF W MAY BE
C       TREATED AS AN EXTRA COLUMN OF R(.,.).
C     W(IWW-N+I) (I=1,2,...,N) ARE USED FOR TEMPORARY STORAGE.
C     W(IWW+I) (I=1,2,...,N) ARE USED FOR TEMPORARY STORAGE.
C     W(IWD+I) (I=1,2,...,N) HOLDS G(I,I) DURING THE CALCULATION.
C     W(IWX+I) (I=1,2,...,N) HOLDS VARIABLES THAT WILL BE USED TO
C       TEST THAT THE ITERATIONS INCREASE THE OBJECTIVE FUNCTION.
C     W(IWA+K) (K=1,2,...,M) USUALLY HOLDS THE RECIPROCAL OF THE
C       LENGTH OF THE K-TH CONSTRAINT, BUT ITS SIGN INDICATES
C       WHETHER THE CONSTRAINT IS ACTIVE.
C
C   
C   AUTHOR:    K. SCHITTKOWSKI,
C              MATHEMATISCHES INSTITUT,
C              UNIVERSITAET BAYREUTH,
C              8580 BAYREUTH,
C              GERMANY, F.R.
C
C   AUTHOR OF ORIGINAL VERSION:
C              M.J.D. POWELL, DAMTP,
C              UNIVERSITY OF CAMBRIDGE, SILVER STREET
C              CAMBRIDGE,
C              ENGLAND
C
C
C   REFERENCE: M.J.D. POWELL: ZQPCVX, A FORTRAN SUBROUTINE FOR CONVEX
C              PROGRAMMING, REPORT DAMTP/1983/NA17, UNIVERSITY OF
C              CAMBRIDGE, ENGLAND, 1983.
C
C
C   VERSION :  2.0 (MARCH, 1987)
C
C
C*************************************************************************
C
      INTEGER MMAX,NMAX,N,LW
      DIMENSION A(MMAX,NMAX),B(MMAX),GRAD(N),G(NMAX,N),X(N),IACT(N),
     1      W(LW),XL(N),XU(N)
      INTEGER M,MEQ,MN,MNN,NACT,IACT,INFO,MAXIT
      DOUBLE PRECISION CVMAX,DIAG,DIAGR,FDIFF,FDIFFA,GA,GB,PARINC,PARNEW       
     1      ,RATIO,RES,STEP,SUM,SUMX,SUMY,SUMA,SUMB,SUMC,TEMP,TEMPA,
     2       VSMALL,XMAG,XMAGR,ZERO,ONE,TWO,ONHA,VFACT
      DOUBLE PRECISION A,B,G,GRAD,W,X,XL,XU
C
C   INTRINSIC FUNCTIONS:   DMAX1,DSQRT,DABS,DMIN1
C
      INTEGER IWZ,IWR,IWW,IWD,IWA,IFINC,KFINC,K,I,IA,ID,II,IR,IRA,
     1     IRB,J,NM,IZ,IZA,ITERC,ITREF,JFINC,IFLAG,IWS,IS,K1,IW,KK,IP,
     2     IPP,IL,IU,JU,KFLAG,LFLAG,JFLAG,KDROP,NU,MFLAG,KNEXT,IX,IWX,
     3     IWY,IY,JL
      LOGICAL LQL,LOWER
C
C   INITIAL ADDRESSES
C
      IWZ=NMAX
      IWR=IWZ+NMAX*NMAX
      IWW=IWR+(NMAX*(NMAX+3))/2
      IWD=IWW+NMAX
      IWX=IWD+NMAX
      IWA=IWX+NMAX
C
C     SET SOME CONSTANTS.
C
      ZERO=0.D+0
      ONE=1.D+0
      TWO=2.D+0
      ONHA=1.5D+0
      VFACT=1.D+0
C
C     SET SOME PARAMETERS.
C     NUMBER LESS THAN VSMALL ARE ASSUMED TO BE NEGLIGIBLE.
C     THE MULTIPLE OF I THAT IS ADDED TO G IS AT MOST DIAGR TIMES
C       THE LEAST MULTIPLE OF I THAT GIVES POSITIVE DEFINITENESS.
C     X IS RE-INITIALISED IF ITS MAGNITUDE IS REDUCED BY THE
C       FACTOR XMAGR.
C     A CHECK IS MADE FOR AN INCREASE IN F EVERY IFINC ITERATIONS,
C       AFTER KFINC ITERATIONS ARE COMPLETED.
C
      DIAGR=TWO
      XMAGR=1.0D-2
      IFINC=3
      KFINC=MAX0(10,N)
C
C     FIND THE RECIPROCALS OF THE LENGTHS OF THE CONSTRAINT NORMALS.
C     RETURN IF A CONSTRAINT IS INFEASIBLE DUE TO A ZERO NORMAL.
C
      NACT=0
      IF (M .LE. 0) GOTO 45
      DO 40 K=1,M
      SUM=ZERO
      DO 10 I=1,N
   10 SUM=SUM+A(K,I)**2
      IF (SUM .GT. ZERO) GOTO 20
      IF (B(K) .EQ. ZERO) GOTO 30
      INFO=-K
      IF (K .LE. MEQ) GOTO 730
      IF (B(K)) 30,30,730
   20 SUM=ONE/DSQRT(SUM)
   30 IA=IWA+K
   40 W(IA)=SUM
   45 DO 50 K=1,N
      IA=IWA+M+K
   50 W(IA)=ONE
C
C     IF NECESSARY INCREASE THE DIAGONAL ELEMENTS OF G.
C
      IF (.NOT. LQL) GOTO 165
      DIAG=ZERO
      DO 60 I=1,N
      ID=IWD+I
      W(ID)=G(I,I)
      DIAG=DMAX1(DIAG,VSMALL-W(ID))
      IF (I .EQ. N) GOTO 60
      II=I+1
      DO 55 J=II,N
      GA=-DMIN1(W(ID),G(J,J))
      GB=DABS(W(ID)-G(J,J))+DABS(G(I,J))
      IF (GB .GT. ZERO) GA=GA+G(I,J)**2/GB
   55 DIAG=DMAX1(DIAG,GA)
   60 CONTINUE
      IF (DIAG .LE. ZERO) GOTO 90
   70 DIAG=DIAGR*DIAG
      DO 80 I=1,N
      ID=IWD+I
   80 G(I,I)=DIAG+W(ID)
C
C     FORM THE CHOLESKY FACTORISATION OF G. THE TRANSPOSE
C     OF THE FACTOR WILL BE PLACED IN THE R-PARTITION OF W.
C
   90 IR=IWR
      DO 130 J=1,N
      IRA=IWR
      IRB=IR+1
      DO 120 I=1,J
      TEMP=G(I,J)
      IF (I .EQ. 1) GOTO 110
      DO 100 K=IRB,IR
      IRA=IRA+1
  100 TEMP=TEMP-W(K)*W(IRA)
  110 IR=IR+1
      IRA=IRA+1
      IF (I .LT. J) W(IR)=TEMP/W(IRA)
  120 CONTINUE
      IF (TEMP .LT. VSMALL) GOTO 140
  130 W(IR)=DSQRT(TEMP)
      GOTO 170
C
C     INCREASE FURTHER THE DIAGONAL ELEMENT OF G.
C
  140 W(J)=ONE
      SUMX=ONE
      K=J
  150 SUM=ZERO
      IRA=IR-1
      DO 160 I=K,J
      SUM=SUM-W(IRA)*W(I)
  160 IRA=IRA+I
      IR=IR-K
      K=K-1
      W(K)=SUM/W(IR)
      SUMX=SUMX+W(K)**2
      IF (K .GE. 2) GOTO 150
      DIAG=DIAG+VSMALL-TEMP/SUMX
      GOTO 70
C
C     STORE THE CHOLESKY FACTORISATION IN THE R-PARTITION
C     OF W.
C
  165 IR=IWR
      DO 166 I=1,N
      DO 166 J=1,I
      IR=IR+1
  166 W(IR)=G(J,I)
C
C     SET Z THE INVERSE OF THE MATRIX IN R.
C
  170 NM=N-1
      DO 220 I=1,N
      IZ=IWZ+I
      IF (I .EQ. 1) GOTO 190
      DO 180 J=2,I
      W(IZ)=ZERO
  180 IZ=IZ+N
  190 IR=IWR+(I+I*I)/2
      W(IZ)=ONE/W(IR)
      IF (I .EQ. N) GOTO 220
      IZA=IZ
      DO 210 J=I,NM
      IR=IR+I
      SUM=ZERO
      DO 200 K=IZA,IZ,N
      SUM=SUM+W(K)*W(IR)
  200 IR=IR+1
      IZ=IZ+N
  210 W(IZ)=-SUM/W(IR)
  220 CONTINUE
C
C     SET THE INITIAL VALUES OF SOME VARIABLES.
C     ITERC COUNTS THE NUMBER OF ITERATIONS.
C     ITREF IS SET TO ONE WHEN ITERATIVE REFINEMENT IS REQUIRED.
C     JFINC INDICATES WHEN TO TEST FOR AN INCREASE IN F.
C
      ITERC=1
      ITREF=0
      JFINC=-KFINC
C
C     SET X TO ZERO AND SET THE CORRESPONDING RESIDUALS OF THE
C     KUHN-TUCKER CONDITIONS.
C
  230 IFLAG=1
      IWS=IWW-N
      DO 240 I=1,N
      X(I)=ZERO
      IW=IWW+I
      W(IW)=GRAD(I)
      IF (I .GT. NACT) GOTO 240
      W(I)=ZERO
      IS=IWS+I
      K=IACT(I)
      IF (K .LE. M) GOTO 235
      IF (K .GT. MN) GOTO 234
      K1=K-M
      W(IS)=XL(K1)
      GOTO 240
  234 K1=K-MN
      W(IS)=-XU(K1)
      GOTO 240
  235 W(IS)=B(K)
  240 CONTINUE
      XMAG=ZERO
      VFACT=1.D+0
      IF (NACT) 340,340,280
C
C     SET THE RESIDUALS OF THE KUHN-TUCKER CONDITIONS FOR GENERAL X.
C
  250 IFLAG=2
      IWS=IWW-N
      DO 260 I=1,N
      IW=IWW+I
      W(IW)=GRAD(I)
      IF (LQL) GOTO 259
      ID=IWD+I
      W(ID)=ZERO
      DO 251 J=I,N
  251 W(ID)=W(ID)+G(I,J)*X(J)
      DO 252 J=1,I
      ID=IWD+J
  252 W(IW)=W(IW)+G(J,I)*W(ID)
      GOTO 260
  259 DO 261 J=1,N
  261 W(IW)=W(IW)+G(I,J)*X(J)
  260 CONTINUE
      IF (NACT .EQ. 0) GOTO 340
      DO 270 K=1,NACT
      KK=IACT(K)
      IS=IWS+K
      IF (KK .GT. M) GOTO 265
      W(IS)=B(KK)
      DO 264 I=1,N
      IW=IWW+I
      W(IW)=W(IW)-W(K)*A(KK,I)
  264 W(IS)=W(IS)-X(I)*A(KK,I)
      GOTO 270
  265 IF (KK .GT. MN) GOTO 266
      K1=KK-M
      IW=IWW+K1
      W(IW)=W(IW)-W(K)
      W(IS)=XL(K1)-X(K1)
      GOTO 270
  266 K1=KK-MN
      IW=IWW+K1
      W(IW)=W(IW)+W(K)
      W(IS)=-XU(K1)+X(K1)
  270 CONTINUE
C
C     PRE-MULTIPLY THE VECTOR IN THE S-PARTITION OF W BY THE
C     INVERS OF R TRANSPOSE.
C
  280 IR=IWR
      IP=IWW+1
      IPP=IWW+N
      IL=IWS+1
      IU=IWS+NACT
      DO 310 I=IL,IU
      SUM=ZERO
      IF (I .EQ. IL) GOTO 300
      JU=I-1
      DO 290 J=IL,JU
      IR=IR+1
  290 SUM=SUM+W(IR)*W(J)
  300 IR=IR+1
  310 W(I)=(W(I)-SUM)/W(IR)
C
C     SHIFT X TO SATISFY THE ACTIVE CONSTRAINTS AND MAKE THE
C     CORRESPONDING CHANGE TO THE GRADIENT RESIDUALS.
C
      DO 330 I=1,N
      IZ=IWZ+I
      SUM=ZERO
      DO 320 J=IL,IU
      SUM=SUM+W(J)*W(IZ)
  320 IZ=IZ+N
      X(I)=X(I)+SUM
      IF (LQL) GOTO 329
      ID=IWD+I
      W(ID)=ZERO
      DO 321 J=I,N
  321 W(ID)=W(ID)+G(I,J)*SUM
      IW=IWW+I
      DO 322 J=1,I
      ID=IWD+J
  322 W(IW)=W(IW)+G(J,I)*W(ID)
      GOTO 330
  329 DO 331 J=1,N
      IW=IWW+J
  331 W(IW)=W(IW)+SUM*G(I,J)
  330 CONTINUE
C
C     FORM THE SCALAR PRODUCT OF THE CURRENT GRADIENT RESIDUALS
C     WITH EACH COLUMN OF Z.
C
  340 KFLAG=1
      GOTO 930
  350 IF (NACT .EQ. N) GOTO 380
C
C     SHIFT X SO THAT IT SATISFIES THE REMAINING KUHN-TUCKER
C     CONDITIONS.
C
      IL=IWS+NACT+1
      IZA=IWZ+NACT*N
      DO 370 I=1,N
      SUM=ZERO
      IZ=IZA+I
      DO 360 J=IL,IWW
      SUM=SUM+W(IZ)*W(J)
  360 IZ=IZ+N
  370 X(I)=X(I)-SUM
      INFO=0
      IF (NACT .EQ. 0) GOTO 410
C
C     UPDATE THE LAGRANGE MULTIPLIERS.
C
  380 LFLAG=3
      GOTO 740
  390 DO 400 K=1,NACT
      IW=IWW+K
  400 W(K)=W(K)+W(IW)
C
C     REVISE THE VALUES OF XMAG.
C     BRANCH IF ITERATIVE REFINEMENT IS REQUIRED.
C
  410 JFLAG=1
      GOTO 910
  420 IF (IFLAG .EQ. ITREF) GOTO 250
C
C     DELETE A CONSTRAINT IF A LAGRANGE MULTIPLIER OF AN
C     INEQUALITY CONSTRAINT IS NEGATIVE.
C
      KDROP=0
      GOTO 440
  430 KDROP=KDROP+1
      IF (W(KDROP) .GE. ZERO) GOTO 440
      IF (IACT(KDROP) .LE. MEQ) GOTO 440
      NU=NACT
      MFLAG=1
      GOTO 800
  440 IF (KDROP .LT. NACT) GOTO 430
C
C     SEEK THE GREATEAST NORMALISED CONSTRAINT VIOLATION, DISREGARDING
C     ANY THAT MAY BE DUE TO COMPUTER ROUNDING ERRORS.
C
  450 CVMAX=ZERO
      IF (M .LE. 0) GOTO 481
      DO 480 K=1,M
      IA=IWA+K
      IF (W(IA) .LE. ZERO) GOTO 480
      SUM=-B(K)
      DO 460 I=1,N
  460 SUM=SUM+X(I)*A(K,I)
      SUMX=-SUM*W(IA)
      IF (K .LE. MEQ) SUMX=DABS(SUMX)
      IF (SUMX .LE. CVMAX) GOTO 480
      TEMP=DABS(B(K))
      DO 470 I=1,N
  470 TEMP=TEMP+DABS(X(I)*A(K,I))
      TEMPA=TEMP+DABS(SUM)
      IF (TEMPA .LE. TEMP) GOTO 480
      TEMP=TEMP+ONHA*DABS(SUM)
      IF (TEMP .LE. TEMPA) GOTO 480
      CVMAX=SUMX
      RES=SUM
      KNEXT=K
  480 CONTINUE
  481 DO 485 K=1,N
      LOWER=.TRUE.
      IA=IWA+M+K
      IF (W(IA) .LE. ZERO) GOTO 485
      SUM=XL(K)-X(K)
      IF (SUM) 482,485,483
  482 SUM=X(K)-XU(K)
      LOWER=.FALSE.
  483 IF (SUM .LE. CVMAX) GOTO 485
      CVMAX=SUM
      RES=-SUM
      KNEXT=K+M
      IF (LOWER) GOTO 485
      KNEXT=K+MN
  485 CONTINUE
C
C     TEST FOR CONVERGENCE
C
      INFO=0
      IF (CVMAX .LE. VSMALL) GOTO 700
C
C     RETURN IF, DUE TO ROUNDING ERRORS, THE ACTUAL CHANGE IN
C     X MAY NOT INCREASE THE OBJECTIVE FUNCTION
C
      JFINC=JFINC+1
      IF (JFINC .EQ. 0) GOTO 510
      IF (JFINC .NE. IFINC) GOTO 530
      FDIFF=ZERO
      FDIFFA=ZERO
      DO 500 I=1,N
      SUM=TWO*GRAD(I)
      SUMX=DABS(SUM)
      IF (LQL) GOTO 489
      ID=IWD+I
      W(ID)=ZERO
      DO 486 J=I,N
      IX=IWX+J
  486 W(ID)=W(ID)+G(I,J)*(W(IX)+X(J))
      DO 487 J=1,I
      ID=IWD+J
      TEMP=G(J,I)*W(ID)
      SUM=SUM+TEMP
  487 SUMX=SUMX+DABS(TEMP)
      GOTO 495
  489 DO 490 J=1,N
      IX=IWX+J
      TEMP=G(I,J)*(W(IX)+X(J))
      SUM=SUM+TEMP
  490 SUMX=SUMX+DABS(TEMP)
  495 IX=IWX+I
      FDIFF=FDIFF+SUM*(X(I)-W(IX))
  500 FDIFFA=FDIFFA+SUMX*DABS(X(I)-W(IX))
      INFO=2
      SUM=FDIFFA+FDIFF
      IF (SUM .LE. FDIFFA) GOTO 700
      TEMP=FDIFFA+ONHA*FDIFF
      IF (TEMP .LE. SUM) GOTO 700
      JFINC=0
      INFO=0
  510 DO 520 I=1,N
      IX=IWX+I
  520 W(IX)=X(I)
C
C     FORM THE SCALAR PRODUCT OF THE NEW CONSTRAINT NORMAL WITH EACH
C     COLUMN OF Z. PARNEW WILL BECOME THE LAGRANGE MULTIPLIER OF
C     THE NEW CONSTRAINT.
C
  530 ITERC=ITERC+1
      IF (ITERC.LE.MAXIT) GOTO 531
      INFO=1
      GOTO 710
  531 CONTINUE
      IWS=IWR+(NACT+NACT*NACT)/2
      IF (KNEXT .GT. M) GOTO 541
      DO 540 I=1,N
      IW=IWW+I
  540 W(IW)=A(KNEXT,I)
      GOTO 549
  541 DO 542 I=1,N
      IW=IWW+I
  542 W(IW)=ZERO
      K1=KNEXT-M
      IF (K1 .GT. N) GOTO 545
      IW=IWW+K1
      W(IW)=ONE
      IZ=IWZ+K1
      DO 543 I=1,N
      IS=IWS+I
      W(IS)=W(IZ)
  543 IZ=IZ+N
      GOTO 550
  545 K1=KNEXT-MN
      IW=IWW+K1
      W(IW)=-ONE
      IZ=IWZ+K1
      DO 546 I=1,N
      IS=IWS+I
      W(IS)=-W(IZ)
  546 IZ=IZ+N
      GOTO 550
  549 KFLAG=2
      GOTO 930
  550 PARNEW=ZERO
C
C     APPLY GIVENS ROTATIONS TO MAKE THE LAST (N-NACT-2) SCALAR
C     PRODUCTS EQUAL TO ZERO.
C
      IF (NACT .EQ. N) GOTO 570
      NU=N
      NFLAG=1
      GOTO 860
C
C     BRANCH IF THERE IS NO NEED TO DELETE A CONSTRAINT.
C
  560 IS=IWS+NACT
      IF (NACT .EQ. 0) GOTO 640
      SUMA=ZERO
      SUMB=ZERO
      SUMC=ZERO
      IZ=IWZ+NACT*N
      DO 563 I=1,N
      IZ=IZ+1
      IW=IWW+I
      SUMA=SUMA+W(IW)*W(IZ)
      SUMB=SUMB+DABS(W(IW)*W(IZ))
  563 SUMC=SUMC+W(IZ)**2
      TEMP=SUMB+.1D+0*DABS(SUMA)
      TEMPA=SUMB+.2D+0*DABS(SUMA)
      IF (TEMP .LE. SUMB) GOTO 570
      IF (TEMPA .LE. TEMP) GOTO 570
      IF (SUMB .GT. VSMALL) GOTO 5
      GOTO 570
    5 SUMC=DSQRT(SUMC)
      IA=IWA+KNEXT
      IF (KNEXT .LE. M) SUMC=SUMC/W(IA)
      TEMP=SUMC+.1D+0*DABS(SUMA)
      TEMPA=SUMC+.2D+0*DABS(SUMA)
      IF (TEMP .LE. SUMC) GOTO 567
      IF (TEMPA .LE. TEMP) GOTO 567
      GOTO 640
C
C     CALCULATE THE MULTIPLIERS FOR THE NEW CONSTRAINT NORMAL
C     EXPRESSED IN TERMS OF THE ACTIVE CONSTRAINT NORMALS.
C     THEN WORK OUT WHICH CONTRAINT TO DROP.
C
  567 LFLAG=4
      GOTO 740
  570 LFLAG=1
      GOTO 740
C
C     COMPLETE THE TEST FOR LINEARLY DEPENDENT CONSTRAINTS.
C
  571 IF (KNEXT .GT. M) GOTO 574
      DO 573 I=1,N
      SUMA=A(KNEXT,I)
      SUMB=DABS(SUMA)
      IF (NACT.EQ.0) GOTO 581
      DO 572 K=1,NACT
      KK=IACT(K)
      IF (KK.LE.M) GOTO 568
      KK=KK-M
      TEMP=ZERO
      IF (KK.EQ.I) TEMP=W(IWW+KK)
      KK=KK-N
      IF (KK.EQ.I) TEMP=-W(IWW+KK)
      GOTO 569
  568 CONTINUE
      IW=IWW+K
      TEMP=W(IW)*A(KK,I)
  569 CONTINUE
      SUMA=SUMA-TEMP
  572 SUMB=SUMB+DABS(TEMP)
  581 IF (SUMA .LE. VSMALL) GOTO 573
      TEMP=SUMB+.1D+0*DABS(SUMA)
      TEMPA=SUMB+.2D+0*DABS(SUMA)
      IF (TEMP .LE. SUMB) GOTO 573
      IF (TEMPA .LE. TEMP) GOTO 573
      GOTO 630
  573 CONTINUE
      LFLAG=1
      GOTO 775
  574 K1=KNEXT-M
      IF (K1 .GT. N) K1=K1-N
      DO 578 I=1,N
      SUMA=ZERO
      IF (I .NE. K1) GOTO 575
      SUMA=ONE
      IF (KNEXT .GT. MN) SUMA=-ONE
  575 SUMB=DABS(SUMA)
      IF (NACT.EQ.0) GOTO 582
      DO 577 K=1,NACT
      KK=IACT(K)
      IF (KK .LE. M) GOTO 579
      KK=KK-M
      TEMP=ZERO
      IF (KK.EQ.I) TEMP=W(IWW+KK)
      KK=KK-N
      IF (KK.EQ.I) TEMP=-W(IWW+KK)
      GOTO 576
  579 IW=IWW+K
      TEMP=W(IW)*A(KK,I)
  576 SUMA=SUMA-TEMP
  577 SUMB=SUMB+DABS(TEMP)
  582 TEMP=SUMB+.1D+0*DABS(SUMA)
      TEMPA=SUMB+.2D+0*DABS(SUMA)
      IF (TEMP .LE. SUMB) GOTO 578
      IF (TEMPA .LE. TEMP) GOTO 578
      GOTO 630
  578 CONTINUE
      LFLAG=1
      GOTO 775
C
C     BRANCH IF THE CONTRAINTS ARE INCONSISTENT.
C
  580 INFO=-KNEXT
      IF (KDROP .EQ. 0) GOTO 700
      PARINC=RATIO
      PARNEW=PARINC
C
C     REVISE THE LAGRANGE MULTIPLIERS OF THE ACTIVE CONSTRAINTS.
C
  590 IF (NACT.EQ.0) GOTO 601
      DO 600 K=1,NACT
      IW=IWW+K
      W(K)=W(K)-PARINC*W(IW)
      IF (IACT(K) .GT. MEQ) W(K)=DMAX1(ZERO,W(K))
  600 CONTINUE
  601 IF (KDROP .EQ. 0) GOTO 680
C
C     DELETE THE CONSTRAINT TO BE DROPPED.
C     SHIFT THE VECTOR OF SCALAR PRODUCTS.
C     THEN, IF APPROPRIATE, MAKE ONE MORE SCALAR PRODUCT ZERO.
C
      NU=NACT+1
      MFLAG=2
      GOTO 800
  610 IWS=IWS-NACT-1
      NU=MIN0(N,NU)
      DO 620 I=1,NU
      IS=IWS+I
      J=IS+NACT
  620 W(IS)=W(J+1)
      NFLAG=2
      GOTO 860
C
C     CALCULATE THE STEP TO THE VIOLATED CONSTRAINT.
C
  630 IS=IWS+NACT
  640 SUMY=W(IS+1)
      STEP=-RES/SUMY
      PARINC=STEP/SUMY
      IF (NACT .EQ. 0) GOTO 660
C
C     CALCULATE THE CHANGES TO THE LAGRANGE MULTIPLIERS, AND REDUCE
C     THE STEP ALONG THE NEW SEARCH DIRECTION IF NECESSARY.
C
      LFLAG=2
      GOTO 740
  650 IF (KDROP .EQ. 0) GOTO 660
      TEMP=ONE-RATIO/PARINC
      IF (TEMP .LE. ZERO) KDROP=0
      IF (KDROP .EQ. 0) GOTO 660
      STEP=RATIO*SUMY
      PARINC=RATIO
      RES=TEMP*RES
C
C     UPDATE X AND THE LAGRANGE MULTIPIERS.
C     DROP A CONSTRAINT IF THE FULL STEP IS NOT TAKEN.
C
  660 IWY=IWZ+NACT*N
      DO 670 I=1,N
      IY=IWY+I
  670 X(I)=X(I)+STEP*W(IY)
      PARNEW=PARNEW+PARINC
      IF (NACT .GE. 1) GOTO 590
C
C     ADD THE NEW CONSTRAINT TO THE ACTIVE SET.
C
  680 NACT=NACT+1
      W(NACT)=PARNEW
      IACT(NACT)=KNEXT
      IA=IWA+KNEXT
      IF (KNEXT .GT. MN) IA=IA-N
      W(IA)=-W(IA)
C
C     ESTIMATE THE MAGNITUDE OF X. THEN BEGIN A NEW ITERATION,
C     RE-INITILISING X IF THIS MAGNITUDE IS SMALL.
C
      JFLAG=2
      GOTO 910
  690 IF (SUM .LT. (XMAGR*XMAG)) GOTO 230
      IF (ITREF) 450,450,250
C
C     INITIATE ITERATIVE REFINEMENT IF IT HAS NOT YET BEEN USED,
C     OR RETURN AFTER RESTORING THE DIAGONAL ELEMENTS OF G.
C
  700 IF (ITERC .EQ. 0) GOTO 710
      ITREF=ITREF+1
      JFINC=-1
      IF (ITREF .EQ. 1) GOTO 250
  710 IF (.NOT. LQL) RETURN
      DO 720 I=1,N
      ID=IWD+I
  720 G(I,I)=W(ID)
  730 RETURN
C
C
C     THE REMAINIG INSTRUCTIONS ARE USED AS SUBROUTINES.
C
C
C********************************************************************
C
C
C     CALCULATE THE LAGRANGE MULTIPLIERS BY PRE-MULTIPLYING THE
C     VECTOR IN THE S-PARTITION OF W BY THE INVERSE OF R.
C
  740 IR=IWR+(NACT+NACT*NACT)/2
      I=NACT
      SUM=ZERO
      GOTO 770
  750 IRA=IR-1
      SUM=ZERO
      IF (NACT.EQ.0) GOTO 761
      DO 760 J=I,NACT
      IW=IWW+J
      SUM=SUM+W(IRA)*W(IW)
  760 IRA=IRA+J
  761 IR=IR-I
      I=I-1
  770 IW=IWW+I
      IS=IWS+I
      W(IW)=(W(IS)-SUM)/W(IR)
      IF (I .GT. 1) GOTO 750
      IF (LFLAG .EQ. 3) GOTO 390
      IF (LFLAG .EQ. 4) GOTO 571
C
C     CALCULATE THE NEXT CONSTRAINT TO DROP.
C
  775 IP=IWW+1
      IPP=IWW+NACT
      KDROP=0
      IF (NACT.EQ.0) GOTO 791
      DO 790 K=1,NACT
      IF (IACT(K) .LE. MEQ) GOTO 790
      IW=IWW+K
      IF ((RES*W(IW)) .GE. ZERO) GOTO 790
      TEMP=W(K)/W(IW)
      IF (KDROP .EQ. 0) GOTO 780
      IF (DABS(TEMP) .GE. DABS(RATIO)) GOTO 790
  780 KDROP=K
      RATIO=TEMP
  790 CONTINUE
  791 GOTO (580,650), LFLAG
C
C
C********************************************************************
C
C
C     DROP THE CONSTRAINT IN POSITION KDROP IN THE ACTIVE SET.
C
  800 IA=IWA+IACT(KDROP)
      IF (IACT(KDROP) .GT. MN) IA=IA-N
      W(IA)=-W(IA)
      IF (KDROP .EQ. NACT) GOTO 850
C
C     SET SOME INDICES AND CALCULATE THE ELEMENTS OF THE NEXT
C     GIVENS ROTATION.
C
      IZ=IWZ+KDROP*N
      IR=IWR+(KDROP+KDROP*KDROP)/2
  810 IRA=IR
      IR=IR+KDROP+1
      TEMP=DMAX1(DABS(W(IR-1)),DABS(W(IR)))
      SUM=TEMP*DSQRT((W(IR-1)/TEMP)**2+(W(IR)/TEMP)**2)
      GA=W(IR-1)/SUM
      GB=W(IR)/SUM
C
C     EXCHANGE THE COLUMNS OF R.
C
      DO 820 I=1,KDROP
      IRA=IRA+1
      J=IRA-KDROP
      TEMP=W(IRA)
      W(IRA)=W(J)
  820 W(J)=TEMP
      W(IR)=ZERO
C
C     APPLY THE ROTATION TO THE ROWS OF R.
C
      W(J)=SUM
      KDROP=KDROP+1
      DO 830 I=KDROP,NU
      TEMP=GA*W(IRA)+GB*W(IRA+1)
      W(IRA+1)=GA*W(IRA+1)-GB*W(IRA)
      W(IRA)=TEMP
  830 IRA=IRA+I
C
C     APPLY THE ROTATION TO THE COLUMNS OF Z.
C
      DO 840 I=1,N
      IZ=IZ+1
      J=IZ-N
      TEMP=GA*W(J)+GB*W(IZ)
      W(IZ)=GA*W(IZ)-GB*W(J)
  840 W(J)=TEMP
C
C     REVISE IACT AND THE LAGRANGE MULTIPLIERS.
C
      IACT(KDROP-1)=IACT(KDROP)
      W(KDROP-1)=W(KDROP)
      IF (KDROP .LT. NACT) GOTO 810
  850 NACT=NACT-1
      GOTO (250,610), MFLAG
C
C
C********************************************************************
C
C
C     APPLY GIVENS ROTATION TO REDUCE SOME OF THE SCALAR
C     PRODUCTS IN THE S-PARTITION OF W TO ZERO.
C
  860 IZ=IWZ+NU*N
  870 IZ=IZ-N
  880 IS=IWS+NU
      NU=NU-1
      IF (NU .EQ. NACT) GOTO 900
      IF (W(IS) .EQ. ZERO) GOTO 870
      TEMP=DMAX1(DABS(W(IS-1)),DABS(W(IS)))
      SUM=TEMP*DSQRT((W(IS-1)/TEMP)**2+(W(IS)/TEMP)**2)
      GA=W(IS-1)/SUM
      GB=W(IS)/SUM
      W(IS-1)=SUM
      DO 890 I=1,N
      K=IZ+N
      TEMP=GA*W(IZ)+GB*W(K)
      W(K)=GA*W(K)-GB*W(IZ)
      W(IZ)=TEMP
  890 IZ=IZ-1
      GOTO 880
  900 GOTO (560,630), NFLAG
C
C
C********************************************************************
C
C
C     CALCULATE THE MAGNITUDE OF X AN REVISE XMAG.
C
  910 SUM=ZERO
      DO 920 I=1,N
      SUM=SUM+DABS(X(I))*VFACT*(DABS(GRAD(I))+DABS(G(I,I)*X(I)))
      IF (LQL) GOTO 920
      IF (SUM .LT. 1.D-30) GOTO 920
      VFACT=1.D-10*VFACT
      SUM=1.D-10*SUM
      XMAG=1.D-10*XMAG
  920 CONTINUE
  925 XMAG=DMAX1(XMAG,SUM)
      GOTO (420,690), JFLAG
C
C
C********************************************************************
C
C
C     PRE-MULTIPLY THE VECTOR IN THE W-PARTITION OF W BY Z TRANSPOSE.
C
  930 JL=IWW+1
      IZ=IWZ
      DO 940 I=1,N
      IS=IWS+I
      W(IS)=ZERO
      IWWN=IWW+N
      DO 940 J=JL,IWWN
      IZ=IZ+1
  940 W(IS)=W(IS)+W(IZ)*W(J)
      GOTO (350,550), KFLAG
      RETURN
      END
