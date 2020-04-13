# Imbalanced-data
 A financial fraud use case.

## Introduzione

In questo progetto esploriamo alcune tecniche per affrontare il problema della classificazione in situazioni di forte sbilancilamento nei dati. Il programma è suddiviso in tre macro passaggi:

1. analisi descrittiva delle variabili;
2. classificazione supervisionata;
3. classificazione non supervisionata.

## Dati

Il dataset considerato è un insieme di 284.807 transazioni di carte di credito all'interno del circuito europeo. Ogni transazione è etichettata come regolare **0** oppure come frode **1**. Il forte sbilanciamento fra le due classi dipende dal fatto che solo 492 transazioni sono di tipo frode.

Ogni transazione è descritta da una variabile in secondi *Time* che rappresenta l'intervallo temporale intercorso con la prima transazione del dataset, dall'importo scambiato o *Amount*  e da 29 variabili che per questione di privacy sono state opportunamente anonimizzate e trasformate tramite PCA. 

Il dataset e ulteriori informazioni posso essere trovate a seguente link: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Analisi descrittiva

In questa fase sono state fatte delle valutazioni di carattere generale sulle variabili che vanno a comporre il dataset. In particolare sono state confrontate per le due classi le distrubioni dei valori di ogni singolo descrittore. Il risultato è visualizzabile con l'aiuto dei grafici *var_dist_xxx.png* della cartella *plots*. E' stato inoltre verificato che le variabili hanno subito effetivamente una trasformazione PCA verificando la completa scorrelazione tramite la matrice di covarianza di cui sono riportati i valori in  *covariance_matrix.png*, sempre all'interno della cartella *plots*.

Non sono stati fatti ragionamenti qualitativi su questi risultati a causa dell'anonimizzazione dei descrittori che ha come conseguenza la perdita di significato nel contesto reale.



## Classificazione supervisionata

Come primo passo sono stati addestrati due semplici modelli di *classificazione lineare* e *regressione logistica*. La performance ottenuta, in condizioni di sbilanciamento, è rissumibile dal valore ROC-AUC che mediamente si attesta su un valore pari a 0.78. 

Successivamente, tramite un sovra-campionamento della classe minoritaria e un sotto-campionamento della classe maggioritaria, sono stati addestrati altri due tipi di modelli: una *random forest* e un *xgboost*. All'interno del codice è stata prevista una funzione per eseguire una *grid-search* dei parametri di entrambi i modelli. Questa funzione restituisce quindi le varie performace di tutti i modelli provati. Fra questi viene scelto in automatico quello che ha ottenuto prestazioni migliori e viene riaddestrato per estrarre tutte le caratteristiche di interesse, come il grafico della curva ROC, la lista delle variabili più importanti nel processo decisionale e la matrice di confusione. Queste informazioni possono essere consultate nei rispettivi grafici nella plots. 

Le performace ottenute dal modello migliore, quasi sempre xgboost, si attestano mediamente intorno ad un valore di ROC-AUC di 0.93.



## Classificazione non supervisionata

In questa fase **non** sono stati forniti ai modelli le etichette che distinguono transazioni normali da transazioni fraudolente. L'unica informazione passata è una stima del rapporto che ci si aspetta fra le due classi che in questo contesto prende il nome di *contaminazione*. 

L'idea dietro questo tipo di approccio è sfruttare lo sbilanciamento dei dati a proprio favore e rilevare le frodi come anomalie di un processo che mediamente è rappresentato da transazioni regolari.

Sono stati addrestrati tre modelli differenti che si basano su logiche differenti.

1. Ensamble: isolation forest.
2. Rete neurale: autoencoder.
3. Modello metrico: Local Outlier Factor.

Osserviamo che il dataset è stato sottocampionato casualmente per addestrare il modello LOF che richiede un grande carico computazionale. Sfruttando le etichette, dopo la fase  training, il programma valuta le permormance, opportunamente aggregate, dei 3 modelli.

Per esempio ha imputato a frodi 928 casi su un totale di 51.070  transazioni coprendo l'85% dei casi effettivi. Questo tipo di risultato è molto utile se pensato in ottica reale, dove il modello è in grado di scremare in maniera consistente i casi prima di essere eventualmente analizzati da qualche agente preposto. Osserviamo che la probabilità estrarre una frode a caso nei dati passa da 0.002 a 0.1.

Nella cartella plots, per ogni modello, sono riportati i valori di anomalia associati a ogni osservazione. I grafici in questione ci aiutano a capire quanto i modelli stanno effettivamente interpretando bene il fenomeno.

Le performace ottenute dal modello migliore, quasi sempre la rete neurale, si attestano mediamente intorno ad un valore di ROC-AUC di 0.90.