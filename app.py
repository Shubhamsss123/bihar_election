import streamlit as st
import pandas as pd
import pickle
from catboost import CatBoostRegressor

# Load the model
with open('catboost_model.pkl', 'rb') as file:
    model1 = pickle.load(file)

def predict_normalized_votes(ac_no, party, alliance):
    # Convert inputs to a DataFrame format similar to training data
    input_data = pd.DataFrame({
        'AC No': [ac_no],
        'Party': [party],
        'Alliance': [alliance],
    })
    # Predict using the trained model
    predicted_normalized_votes = model1.predict(input_data)
    return predicted_normalized_votes[0]

# Streamlit app
st.title('Normalized Votes Prediction')

# Input fields
# ac_no = st.number_input('AC No.', min_value=1, max_value=243, step=1)
ac_no=st.selectbox('AC No.',[str(i) for i in range(1,244)])

parties = ['JD(U)', 'INC', 'JAPL', 'BSP', 'NOTA', 'LJP', 'BPL', 'TPLRSP', 'JNSNGHDL', 'IND', 'BJP', 'BYPP', 'FJKSP', 'RLSP', 'JDR', 'NCP', 'JMBP', 'BJNJGD', 'BHRTSBLP', 'JPS', 'LKSHPLK', 'RJBP', 'BRRTD', 'BLND', 'BAHUMP', 'RJD', 'JRJP', 'ASPKR', 'VCSMP', 'LAJSP', 'PSS', 'RMGP', 'APKSP', 'CPI(ML)(L)', 'AIMIM', 'PPID', 'JAC', 'JANADIP', 'RSWD', 'VSIP', 'AIMF', 'JGHTP', 'HSP', 'YKP', 'BNDl', 'RTJPS', 'BVDU', 'CPIM', 'RJnJnP', 'SWAP', 'BMUP', 'BAJVP', 'BJKVP', 'JP', 'AKHDBRYVP', 'JTVP', 'SHS', 'NPEP', 'BGMP', 'RJJM', 'RSSCMJP', 'AIFB', 'RJSPS', 'RPPRINAT', 'SAP', 'PBLBRP', 'KPOI', 'RSTJLKPS', 'INL', 'SJDD', 'RPI', 'VPI', 'KSLJNP', 'PBI', 'RUC', 'NJANP', 'LGJNPSCL', 'SWARAJ', 'HSAP', 'CPI', 'SANVP', 'BLRP', 'JMVP', 'JNSHKVPD', 'BRCTP', 'JTLP', 'JKiP', 'JmtP', 'AKBMP', 'SatBP', 'PECP', 'BhNP', 'RSTWDMNRTP', 'SANKISVP', 'RPIRASH', 'JHP', 'RTSMJD', 'JD(S)', 'RVNP', 'VBA', 'LPSP', 'MKDVP', 'BJJD', 'APADP', 'AMJNMTP', 'BMF', 'SDPI', 'AngSP', 'BHJJP', 'LJD', 'RAUNTP', 'AIMEIEM', 'HAMS', 'RTORP', 'SJPB', 'ACDP', 'AJPR', 'SBSP', 'RJANSWP', 'RaJPa', 'AZSP', 'JMM', 'SUCI', 'BTMSP', 'BJKD', 'AMiP', 'AAM', 'SAMDAL', 'MZEKP', 'RSPS', 'RJPD', 'RJKPS', 'WAP', 'mimm', 'LOKJANP', 'RSTJNVKSP', 'BHRTLKCHTP', 'LCD', 'AAAAP', 'GJNP', 'BHRTSYTKP', 'JKM', 'RPI(A)', 'GJKP', 'APoI', 'SRSMJNP', 'ManJP', 'AKNDBHRJNP', 'JSVP', 'BBC', 'SBSPSP', 'BHAIP', 'GGP', 'RTPP', 'BJNP', 'RTMNSWP', 'SWSP', 'BRTYLKNYKP', 'RSTSAHP', 'BKKMP', 'SHtP', 'RGPP', 'PRAJAP', 'SPAKP', 'SSD', 'SAAFP', 'YBS', 'AJPS', 'YGKRTDL', 'JPJD', 'SP(I)', 'NTP', 'BDlP', 'BSSPA', 'LSMP', 'SaBP', 'LJVM', 'BBMP', 'NJP', 'pms', 'BIP', 'BHULKD', 'PMPT', 'SPL', 'ABMVMP', 'ASDHP', 'PMP', 'ANC', 'LSSP', 'BRMNP', 'LSWP', 'RJPty', 'BhAmAP', 'NFDEP', 'LSD', 'BKVP', 'JASP', 'MADP', 'HVD', 'RDU', 'RWJS', 'RTJHPR', 'AKBHAD', 'RSTSWNTP', 'RSMD', 'AHFBK', 'RTSHD', 'JRVP', 'BaSaPa', 'MOSP', 'HNDUAWMMNC', 'RJCHD', 'SHD', 'BLSP', 'SP', 'SKLP', 'GJDS', 'BED', 'RAJPA', 'GAAP', 'CPM', 'NJPI', 'JKNPP', 'AVIRP', 'BKP', 'ABHKP', 'AAHPTY', 'RPP', 'RAVP', 'BJHP', 'BNDL', 'RSHJP', 'HND', 'BYPD', 'BHNP', 'VIP', 'AMIP', 'BAJAP', 'SABP', 'ABHM', 'BMTRP', 'HCP', 'KSVP', 'TNRMPI', 'BDBP', 'BVM', 'RKSP', 'LKJP', 'SOJP', 'AICP', 'IUML', 'NATP', 'IEMC', 'MANJP', 'LD', 'MVM', 'DSP', 'AIFB(S)', 'CPI(M (L)L)', 'KSJP', 'JDP', 'BJND', 'LAD', 'SASAPT', 'BHASP', 'NALP', 'KS', 'MCPI', 'BNSKP', 'RYP', 'BJPART Y', 'ABRS', 'BDLP', 'RGD', 'PRRP', 'KVD', 'RSADP', 'NADP', 'BDEP', 'RAIP', 'RAHM', 'LTSD', 'VKAM', 'AAMJP', 'PMS', 'SABJAN', 'IPTY', 'JHD', 'RSP', 'RJPTY', 'HMSP', 'ABJS', 'RSJNP', 'C(S)', 'JHSPT', 'ATBP', 'AD', 'MAJP', 'NYP', 'SAKD', 'RSPTY', 'RJNJNP', 'HKRD', 'IPty', 'RMEP', 'DKP', 'LKSE', 'SJP(R)', 'ABAS', 'IJP', 'RKJP', 'RASED', 'GVIP', 'INGP', 'SMBHP', 'BEP(R)', 'MUL', 'MCPI(S)', 'BSA', 'BMP', 'IJK', 'RSP(S)', 'SLP(L)', 'ABSP', 'JVM', 'LM', 'RWS', 'ABAPSMP', 'BHAJP', 'AIBJRBSNC', 'PMSP', 'ABDBM', 'AJSP', 'SWJP', 'SHSP', 'LS', 'AP', 'BSP(K)']
party = st.selectbox('Party', parties)

alliances = ['NDA', 'MGB', 'Other', 'GDSF', 'Unknown']
alliance = st.selectbox('Alliance', alliances)

if st.button('Predict'):
    if alliance == 'Unknown':
        alliance = None
    result = predict_normalized_votes(ac_no, party, alliance)
    st.success(f'Predicted Normalized Votes: {result:.4f}')

st.info('Note: This prediction is based on the trained CatBoost model.')