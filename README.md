# Projeto ButterFly Effect

## Funcionalidades de cada algoritmo

* **cropped_black_surface.py** recorta parte de interesse de uma Imagem
* **cropped.py** remove fundo a partir de uma segmentation
* **dataset_create.py** copia imagens para dataset/ e divide elas em pastas com respectivas classes
* **hog_descriptor.py** descritor HOG (tanto de teste e aprendizado)
* **lbp_descriptor.py** descritor LBP (tanto de teste e aprendizado)
* **less_red_mask.py** substitui pixels que não são nem pretos e nem brancos
* **normalize_images.py** normaliza imagens em mesma largura e altura
* **pls_butterfly_hog.py** PLS para o descritor HOG
* **pls_butterfly_lbp.py** PLS para o descritor LBP
* **resize_images.py** redimensiona imagens em mesma largura

## Ordem de execução dos algoritmos

Para refinarmos as imagens e criarmos o nosso dataset funcional devemos executar os algoritmos nessa ordem:

```sh
python less_red_mask.py
python cropped_black_surface.py
python cropped.py
# antes de redimensionar faça rotações nas imagens
# para padronizar as proporções
python resize_images.py
python normalize_images.py
python dataset_create.py
# depois é só selecionar as respectivas imagens
# para teste e aprendizado
```

## Resultados e Parâmetros

### Somente HOG

#### Teste 6 -- ( 60-40 ) v1.0

* orientations = 8
* pixels_per_cell = (20, 20)
* cells_per_block = (1, 1)
* transform_sqrt = True
* block_norm = "L1"

ACC = 225
TOTAL = 332

### Somente LBP

#### Teste 3 -- ( 60-40 ) v1.0

* numPoints = 48
* radius = 16

ACC = 248
TOTAL = 332

### LBP-HOG

#### Teste 4 -- ( 60-40 ) v1.1

* numPoints = 72
* radius = 24

* orientations = 8
* pixels_per_cell = (20, 20)
* cells_per_block = (1, 1)
* transform_sqrt = True
* block_norm = "L1"

ACC = 255
TOTAL = 336
