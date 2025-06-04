# 🔸 1. Simulação de Dados (com valores faltantes e outliers)
dados = [
    {'idade': 25, 'salario': 5000, 'classe': 'A'},
    {'idade': 27, 'salario': 5200, 'classe': 'A'},
    {'idade': 29, 'salario': 5100, 'classe': 'B'},
    {'idade': 22, 'salario': 4950, 'classe': 'B'},
    {'idade': None, 'salario': 5050, 'classe': 'A'},
    {'idade': 120, 'salario': 50000, 'classe': 'B'},  # outlier
    {'idade': 26, 'salario': 5000, 'classe': 'A'},
    {'idade': 24, 'salario': 5100, 'classe': 'B'},
    {'idade': 28, 'salario': 5150, 'classe': 'A'},
    {'idade': None, 'salario': 5200, 'classe': 'B'}
]

# 🔸 2. Separar as colunas
idades = [item['idade'] for item in dados if item['idade'] is not None]
salarios = [item['salario'] for item in dados]

# 🔸 3. Calcular Média (para idade)
media_idade = sum(idades) / len(idades)

# 🔸 4. Calcular Mediana (para salário)
salarios_ordenados = sorted(salarios)
n = len(salarios_ordenados)
if n % 2 == 0:
    mediana_salario = (salarios_ordenados[n//2 - 1] + salarios_ordenados[n//2]) / 2
else:
    mediana_salario = salarios_ordenados[n//2]

# 🔸 5. Imputação dos Dados Faltantes
for item in dados:
    if item['idade'] is None:
        item['idade'] = media_idade
    if item['salario'] is None:
        item['salario'] = mediana_salario

# 🔸 6. Calcular Moda (para classe)
classes = [item['classe'] for item in dados]
frequencias = {}
for c in classes:
    frequencias[c] = frequencias.get(c, 0) + 1

moda_classe = max(frequencias, key=frequencias.get)

print(f"🔹 Média das idades: {media_idade:.2f}")
print(f"🔸 Mediana dos salários: {mediana_salario}")
print(f"⭐ Moda das classes: {moda_classe}")

# 🔸 7. Normalização simples (escala 0 a 1)
idades = [item['idade'] for item in dados]
salarios = [item['salario'] for item in dados]

min_idade, max_idade = min(idades), max(idades)
min_salario, max_salario = min(salarios), max(salarios)

for item in dados:
    item['idade_norm'] = (item['idade'] - min_idade) / (max_idade - min_idade)
    item['salario_norm'] = (item['salario'] - min_salario) / (max_salario - min_salario)

# 🔸 8. Simulação de Classificação com KNN (k=3)
# Dados de teste (idade=26, salario=5100)
teste = {'idade': 26, 'salario': 5100}

# Normalizar dados de teste
teste['idade_norm'] = (teste['idade'] - min_idade) / (max_idade - min_idade)
teste['salario_norm'] = (teste['salario'] - min_salario) / (max_salario - min_salario)

# Calcular distância euclidiana para todos os dados
def distancia(a, b):
    return ((a['idade_norm'] - b['idade_norm']) ** 2 + (a['salario_norm'] - b['salario_norm']) ** 2) ** 0.5

distancias = []
for item in dados:
    d = distancia(item, teste)
    distancias.append((d, item['classe']))

# Ordenar pelas menores distâncias
distancias.sort(key=lambda x: x[0])

# Selecionar os 3 vizinhos mais próximos
k = 3
vizinhos = distancias[:k]

# Contar a frequência das classes dos vizinhos
contagem = {}
for _, classe in vizinhos:
    contagem[classe] = contagem.get(classe, 0) + 1

# Escolher a classe mais frequente (moda)
classe_predita = max(contagem, key=contagem.get)

print(f"\n🔍 Vizinhos mais próximos: {vizinhos}")
print(f"🎯 Classe predita para o teste: {classe_predita}")
