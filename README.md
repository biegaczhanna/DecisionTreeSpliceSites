# DecisionTreeSpliceSites
Drzewo decyzyjne w zadaniu klasyfikacji miejsc rozcięcia w sekwencji DNA.

# Oryginalna treść polecenia
Istnieją dwa rodzaje miejsc rozcięcia sekwencji kodującej białko: donory i akceptory. Ich odnalezienie otwiera drogę do znalezienia eksonów, czyli sekwencji kodujących białka.

Należy zaimplementować klasyfikator, następnie przeprowadzić jego trening i testowanie na 2 problemach:
- szukanie donorów,
- szukanie akceptorów.
Każdy ze zbiorów danych należy rozdzielić na trenujący i testujący lub zastosować walidację krzyżową. Zaimplementowany klasyfikator należy przebadać (wykonać eksperymenty).

Jeżeli chodzi o dane to w tym pliku znajdują się przykłady donorów, a w tym pliku przykłady akceptorów. W pierwszej linii każdego z nich napisano, na której pozycji (licząc litery od lewej strony) we fragmentach sekwencji jest granica pomiędzy intronem a eksonem. Dana ta jest zbędna dla klasyfikatora - może jednak pomóc badaczowi w interpretacji wyników. Dalej w pliku występują parami: linia określająca czy jest to przykład pozytywny (1) czy negatywny (0) oraz sam przykład czyli sekwencja DNA. Przykłady negatywne to takie, które częściowo wyglądają jak miejsca rozcięcia, ale nimi nie są.

# Dobór parametrów
Za pomocą metody grid search przeanalizowano szeroki zakres doboru parametrów dla drzewa decyzyjnego. Poniżej przedstawione są wyniki uzyskane na zbiorze danych zawierających **donorów**. 
Dotychczas najlepsze parametry to: 
- Depth=10  
- MinSamples=70 
- MinGain=0.0
- TrainSetSize=0.8

Drzewo o tak skonfigurowanych parametrach zostało przeanalizowane za pomocą metody walidacji krzyżowej i uzyskało następujące wyniki:
- Mean Accuracy: 0.8893
- Mean Recall: 0.6640
- Mean Precision: 0.7400
- Mean Confusion Matrix:
    [[1468.66666667   88.66666667]
    [ 125.          247.        ]]

To drzewo jest też zapisane w pliku trees/best_donor_tree.txt