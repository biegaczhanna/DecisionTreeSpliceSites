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
- Depth=20  
- MinSamples=10 
- MinGain=0.0
- TrainSetSize=0.8

Drzewo o tak skonfigurowanych parametrach zostało przeanalizowane za pomocą metody walidacji krzyżowej i uzyskało następujące wyniki:
- Mean Accuracy: 0.9285
- Mean Recall: 0.8306
- Mean Precision: 0.8329
- Mean Confusion Matrix:
    [[1317.66666667   62.33333333]
    [  63.          309.        ]]
To drzewo jest też zapisane w pliku trees/best_donor_tree.txt

Za to na zbiorze danych zawierającym akceptory, znalezione najlepsze parametry to:
- Depth=40
- MinSamples=110
- MinGain=0.0
- TrainSetSize=0.8

Wynika to z faktu, że sekwencje DNA w tym zbiorze są znacznie dłuższe, składają się z 90 liter, podczas gdy te ze zbioru donorów jedynie z 15. Dlatego potrzebna jest większa głębokość drzewa i większa liczba próbek, aby móc poprawnie sklasyfikować sekwencje.

```bash
==================== Processing Donors ====================
---> Running Grid Search
Best Accuracy found: 0.9503
Parameters: Depth=5, MinSamples=2, MinGain=0.01, TrainSetSize=0.8

---> Cross Validation for best parameters
Mean Accuracy: 0.9479
Mean Recall: 0.8817
Mean Precision: 0.8749
Mean Confusion Matrix:
[[1332.66666667   47.33333333]
 [  44.          328.        ]]

==================== Processing Acceptors ====================
---> Running Grid Search
Best Accuracy found: 0.8965
Parameters: Depth=20, MinSamples=70, MinGain=0.01, TrainSetSize=0.8

---> Cross Validation for best parameters
Mean Accuracy: 0.8906
Mean Recall: 0.7142
Mean Precision: 0.7176
Mean Confusion Matrix:
[[1452.66666667  104.66666667]
 [ 106.33333333  265.66666667]]
```