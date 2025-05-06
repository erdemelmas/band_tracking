import streamlit as st

# Başlık
st.title('İsim Soyisim Alma Formu')

# Form alanları
isim = st.text_input('İsim')
soyisim = st.text_input('Soyisim')

# Gönder butonu
if st.button('Gönder'):
    st.write(f'İsim: {isim}, Soyisim: {soyisim}')