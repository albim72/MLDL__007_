def czytaj_liste(lista):
    for i,j in enumerate(lista):
        print(f'element listy {i+1}, numer filii: {j}')

def czytaj_slownik(slownik):
    for x,y in slownik.item():
        print(f'element słownika: klucz -> {x}, wartośc -> {y}')
