import streamlit as st

# Initialize session state
if "menu_choice" not in st.session_state:
    st.session_state.menu_choice = "Strona główna"

def main():
    st.title("Weterynarz")

    # Sidebar menu
    st.sidebar.title("Menu")
    button_home = st.sidebar.button("Strona główna")
    button_recognize = st.sidebar.button("Rozpoznaj psiaka")
    button_discover = st.sidebar.button("Odkryj psiaka")

    # Main page content
    if button_home:
        st.session_state.menu_choice = "Strona główna"
    elif button_recognize:
        st.session_state.menu_choice = "Rozpoznaj psiaka"
    elif button_discover:
        st.session_state.menu_choice = "Odkryj psiaka"

    # Show content based on selected option
    if st.session_state.menu_choice == "Strona główna":
        st.write("Witaj na stronie głównej!")
        st.write("Pies jest jednym z najbardziej oddanych i lojalnych zwierząt domowych. Towarzyszą ludziom od tysięcy lat i pełnią różnorodne role, od stróżów po przyjaciół i terapeutów. Wiele ras psów ma swoje unikalne cechy fizyczne i temperament, co czyni je fascynującymi istotami do poznawania.")
        st.write("Na tej stronie możesz skorzystać z funkcji rozpoznawania rasy psa na podstawie zdjęcia oraz odkrywać ciekawostki na temat różnych ras.")
        st.image("mainpage.jpg", caption='Obrazek na stronie głównej', use_column_width=True)

    return st.session_state.menu_choice

if __name__ == "__main__":
    main()
