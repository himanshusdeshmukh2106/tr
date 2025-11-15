in()
ma:
    "__main__" == me____na

if 
"="*80)
    print(one! 🎉")   print("D0)
 + "="*8\n" nt("ri p
   ")
    e']:.1%})dencConfist['ce: {late (confident enoughonfiden is not celod(f"   M  printP")
      SKIHOLD/ATION: ⚪ RECOMMENDint(f"\n        prelse:
  )
  DOWN"l go  price wil%} confidentnce']:.1nfideest['Co{lat  Model is nt(f"         pri")
RT SELL/SHOATION:COMMEND\n🔴 REt(f"in  pr     :
  == 'SELL'al']st['Signateif lP")
    elll go Uent price wi confide']:.1%}nfidencatest['Codel is {lt(f"   Moprin        N: BUY")
MENDATIOn🟢 RECOMt(f"\        prinBUY':
== 'al'] st['Sign  if late
    
  l']}")gna'Silatest["  Signal: {int(fpr1%}")
    fidence']:.st['Con{lateence: f"  Confid
    print(")tion']:.3f}redicst['Ption: {lateredic Pnt(f" ")
    priClose']:.2f}test['rice: ₹{laClose P  rint(f"")
    pst.name} Date: {laterint(f"   p")
   Data Point:f"\nLatest
    print(iloc[-1]t = results.tes 
    la   0)
rint("="*8    pIONS")
RECOMMENDATTRADING rint("   p)
 + "="*80int("\n" ns
    prdationg recommenradi t Show   #  
   
_file}"){outputns saved to  Predictioint(f"\n✓le)
    pr_fiututp(oo_csv results.t   s.csv"
ediction_preliance_file = "r outputesults
    Save r
    #s)
    ltignals(resuyze_sdictor.anal   pree
     # Analyz
    
7)ld=0.hoence_thres, confidict(df.pred= predictorsults 
    rensredictio # Make p   E)
    
_FILDATApare_data(nd_preoad_adictor.l = pre  dfdata
   prepare oad and 
    # L)
   ePredictor(or = Reliancdicttor
    predic preializenit I  #)
    
  "*80print("=  CTION")
  DICE STOCK PRE"RELIAN    print(0)
="*8    print("""
w"loorkfrediction w"Main p""n():
    


def mairesults   return 
             )
()ngstri(recent.to_     print]
       gnal']', 'Sience, 'ConfidPrediction'['Close', 'l(10)[.taigh_conf recent = hi        )
   ignals:"cent Sost Rerint(f"\nM    p
        alssigncent Show re    #  
               ")
    .sum():,}LL') 'SE'] ==Signalf['high_con{("  SELL:  print(f     ")
      ).sum():,} 'BUY' ==l']conf['Signa{(high_(f"  BUY: intpr          ,}")
  conf):h_en(hignals: {lnfidence Sig\nHigh Corint(f"      p> 0:
      conf) en(high_    if lHOLD']
    l'] != 'sults['Signaresults[regh_conf =     hignals
    ence si confidgh# Hi         
   ")
    .3f}():ence'].max['Confid: {results Maxf" int(
        pr}")in():.3fnfidence'].mlts['Coresu: {  Minrint(f"  p")
      ian():.3f}ence'].medlts['Confidresuedian: {print(f"  M  }")
      ean():.3fnfidence'].msults['Co"  Mean: {re  print(f     
 s:")atisticonfidence St"\nC(fnt      priics
  ce statistonfiden       # C
        
 t:.1f}%)")pcnt:,} ({ {cou{signal}:t(f"   prin   00
        s) * 1esult/ len(rct = count           p():
  mscounts.ite signal_unt in signal, co for")
       ion:Distributnal Sig(f"\nint   prs()
     countl'].value_s['Signas = resultl_countsigna     ution
   istribignal d      # S         

 0)"="*8  print(  ")
    ON ANALYSISTIPREDICprint(")
        " + "="*80t("\nin pr
       ""results"tion edic"Analyze pr   ""    lts):
 suregnals(self, f analyze_si  
    delts
  n resu       retur        
 '
] = 'SELL 'Signal'threshold),confidence_1 - ctions < ([prediocresults.l        Y'
'] = 'BUld, 'Signal_threshoenceidonf> credictions lts.loc[pesu        rD'
] = 'HOLSignal'ts['   resul
     e signalsater      # Gen
  
        ions)ict, 1 - predctionsmum(predi = np.maxifidence']['Conesults     r
   edictionsion'] = prPredicts['  result
      copy()indices].loc[valid_lts = df.isu       reaframe
  results dat  # Create  
          latten()
  ns.f= predictioedictions pr        se=0)
verbo_sequences, predict(Xf.model.s = seltionpredic
        redict  # P  
         ces")
   ces)} sequenlen(X_sequenreated {"✓ C print(f          

     quences).array(X_seuences = np     X_seq
         )
  ces.append(i_indialid           vh:i])
 ence_lengti-self.sequed[ppend(X_scalsequences.a          X_d)):
  aleth, len(X_sclengence_sequange(self.  for i in r         
  ]
   ndices = [   valid_i  s = []
   quence X_se
       ces sequen Create
        #        atures])
df[self.fensform(f.scaler.tra seld =  X_scales
      le feature   # Sca   
     pna()
      = df.dro     df  
  # Drop NaN      
 
        ")tions...aking predicrint("\nM
        p"""dataew ns on ndictiopreMake       """
  shold=0.7):ence_threid, conflf, dfpredict(se 
    def     df
     return     
   ] / 24)
   f['Hour' np.pi * d= np.cos(2 *] f['Hour_Cos'       d24)
 'Hour'] / f[ np.pi * d(2 *'] = np.sin'Hour_Sin     df[   te
ndex.minu'] = df.inutedf['Mi
        ex.hour= df.ind'Hour'] df[
        tures feaime       # T
        
 ']f['Close_20']) / d'Low'] - df[se'Clo= (df['] stance_Low     df['Di   
['Close']ose']) / df df['Cl0'] -h_2'Hig (df[nce_High'] =['Dista
        df.min()rolling(20)].Low' = df['ow_20'] df['L()
       g(20).maxigh'].rollin] = df['H0'gh_2      df['Hi levels
  ice       # Pr  
 10)
       1e-'] +Volume_MA] / (df['e'df['Volumatio'] = me_R'Volu       df[)
 ).mean(ling(20e'].rollum = df['VoMA']me_f['Volu      dvolume()
  .on_balance_'])Volume df[''Close'],dicator(df[InnceVolumee.OnBalaa.volum'OBV'] = t df[  lume
            # Vo
       
  ge()rane_true_averag'Close']).f[, d['Low'] df(df['High'],angeAverageTrueRility.= ta.volatf['ATR'] 
        dClose']]) / df['er'- df['BB_LowUpper'] BB_] = (df['th'B_Widdf['B()
        er_lbandingbollower'] = bb.['BB_L  df
      nd()hbainger_ollr'] = bb.bdf['BB_Uppe])
        (df['Close'ingerBandsity.Boll= ta.volatil   bb      tility
   # Vola         
    )
.adx(f['Close'])'Low'], d, df[High']dicator(df['trend.ADXInta.'] = f['ADX d()
       _signal = macd.macdACD_Signal'] df['M)
       d(= macd.macf['MACD']       dlose'])
  CD(df['CMA = ta.trend.       macdend
      # Tr        
   _index()
ow]).money_flume''], df['Vol df['Close['Low'],], df['High'tor(dfme.MFIIndica ta.volu'] =   df['MFI
     si()dow=14).re'], winf['Closdicator(dm.RSIIn ta.momentuSI'] =       df['Romentum
 
        # M    ()
    ndicatorma_iw=period).eindo], wf['Close'Indicator(dend.EMAd}'] = ta.trA_{perioEMf[f'      d)
      cator(od).sma_indiindow=peri, w']sedf['Cloator(.SMAIndic = ta.trendd}']{perio df[f'SMA_
           20, 50]:10, [5, in iod  per     for   verages
 # Moving A       
   ge()
     ].pct_chanlume' df['Voge'] =_ChanmeVoludf[')
        ge(e'].pct_chan['Closrns'] = df    df['Retuasic
           # B
        ")
 s...featuret("Adding  prin     """
   trainingas features amedd s"""A
        f, df):eleatures(sd_f ad    def
    
  return df   
      s(df)
     ature self.add_fe  df =    )
  iningame as tra (suresdd feat# A
        
        ows")n(df)} rleaded {"✓ Lo    print(f            
me']
 'VoluClose', 'Low', 'en', 'High',['Opmns = olu  df.c
      )place=Trueetime', in'datx(_indef.set      d
  me'])df['datetidatetime('] = pd.to_tetime   df['da     ilepath)
.read_csv(f = pd df
       # Load data    
        
    ath}...")from {filepading data (f"\nLo     print   ""
ta" new daprepare""Load and     "
    ):ilepath(self, fare_dataad_and_preplo  
    def g
       traininme as0  # Sah = 3gtequence_len  self.s             
res")
 eatuatures)} ff.fes {len(selxpect Model erint(f"✓
        ped")s loadd featureer ant(f"✓ Scal  prin     s.pkl")
 / "featureIR _Dd(MODELblib.loas = joref.featu   sell")
     r.pkscale_DIR / "(MODEL.loadoblibr = jcale   self.s
     atureser and feoad scal        # L
      ath}")
  del_p from {mol loaded Modeprint(f"✓       th)
 model_pa.load_model(ras.modelsdel = tf.kef.mo      selel
   Load mod        #
        
del...")moing nt("Load     pri"
   "objects"ssing reproceodel and pined m tra""Load     ":
   h5")t_model.alanced/besance_breli="thlf, model_pa __init__(se def 
   s"""
   prediction make ad model and  """Lo
  ctor:ediancePrReliss "


cla.csvull_yeardata_5min_fnce_lia_FILE = "re")
DATAbalanced"reliance_IR = Path(DEL_Ds
MO')

# Pathngs('ignorelterwarnifi
warnings.arnings
import wort Pathlib impm pathro
f taportblib
imt jo
imporflow as tforimport tenss pd
t pandas aorp
imp n ast numpy"

impor" Data
"n Relianceions odict to Make Predelned Morai
Use T"""