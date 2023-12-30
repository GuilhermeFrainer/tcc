import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sktime.param_est.stationarity import StationarityKPSS
from sktime.forecasting.model_evaluation import evaluate
from sktime.split import temporal_train_test_split, ExpandingWindowSplitter
from sktime.forecasting.base import ForecastingHorizon


def evaluate_pipeline(
        forecaster,
        df: pd.DataFrame,
        steps_ahead=1,
        test_size=79,
):
    """
    Padroniza o processo repetitivo de chamar a função 'evaluate'

    Parameters
    ----------

    forecaster: sktime BaseForecaster descendant (concrete forecaster) 
        Forecaster a ser utilizado nas previsões
    
    df: pd.DataFrame
        DataFrame com variáveis endógena e exógenas.
        Presume que variável endógena esteja na coluna 'ipca'.

    steps_ahead: int (default=1)
        Número de passos à frente para prever.
        
    test_size: int (default=79)
        Tamanho da amostra usada para testes.
        Na prática, para os propósitos do TCC, vai ser sempre 79.

    Returns
    -------
    y_train: pd.Series
        Série do trecho da variável endógena usada para treinar o modelo

    y_test: pd.Series
        Série do trecho da variável endógena usada para testar o modelo

    y_pred: pd.DataFrame
        Série de previsões extraída com a função 'extract_y_pred'.
    """
    y, X = extract_X_and_y(df)
    y_train, y_test, _, _ = temporal_train_test_split(y, X, test_size=test_size)
    fh = ForecastingHorizon(steps_ahead, is_relative=True)
    cv = ExpandingWindowSplitter(fh=fh, initial_window=y_train.size)
    eval_df =  evaluate(
        forecaster=forecaster,
        cv=cv,
        y=y,
        X=X,
        strategy="refit",
        return_data=True
    )
    y_pred = extract_y_pred(eval_df)
    return y_train, y_test, y_pred


def extract_y_pred(df: pd.DataFrame) -> pd.Series:
    """
    Extrai previsões do DataFrame retornado pela função 'evaluate' do sktime.
    Necessário pois a função retorna as previsões formatadas de um jeito esquisito.

    Parameters
    ----------

    df: pd.DataFrame
        DataFrame retornado pela função 'evaluate' do sktime.
        Função precisa ter sido chamada com o parâmetro 'return_data=True'.

    Returns
    -------
    pd.Series
        Série contendo apenas previsões.
    """
    y_pred = pd.Series([value.iloc[0] for _, value in df['y_pred'].items()])
    period_list = [entry.index for entry in df['y_pred']]
    period_array = pd.arrays.PeriodArray(period_list, dtype=pd.PeriodDtype("M"))
    period_list2 = [entry[0] for entry in period_array]
    y_pred.index = pd.PeriodIndex(period_list2, freq="M")
    y_pred.index.name = "month"
    y_pred.name = "ipca" # Útil apenas no meu caso específico
    return y_pred


def extract_X_and_y(df: pd.DataFrame):
    """
    Separa variável dependente (ipca) de variáveis independentes.

    Parameters
    ----------

    df: pd.DataFrame
        DataFrame com coluna 'ipca'

    Returns
    -------
    pd.Series
        Série de y

    pd.DataFrame
        DataFrame com variáveis independentes
    """
    y = df['ipca']
    X = df.drop(columns=['ipca'])
    return y, X


def parse_date(s: str) -> str:
    """
    Parseia datas no formato mmm/yy para formato ISO

    Parameters
    ----------

    s: str
        string contendo data no formato mmm/yy

    Returns
    -------
    str
        string no formato ISO de data (yyyy-mm-dd)
    """
    month_dict = {
        'jan': 1,
        'fev': 2,
        'mar': 3,
        'abr': 4,
        'mai': 5,
        'jun': 6,
        'jul': 7,
        'ago': 8,
        'set': 9,
        'out': 10,
        'nov': 11,
        'dez': 12,
    }

    try:
        month, year = s.split("/")
    # Caso não haja '/'
    except ValueError:
        return s
    month = month_dict[month]
    return f"20{year}-{month:02d}-01"


def plot_series(*series, labels=None, figsize=plt.figaspect(0.25)):
    """
    Serve para replicar a função 'plot_series' do sktime,
    mas sem o problema de mudar a escala de cada série.

    Parameters
    ----------

    series: pd.Series or iterable of pd.Series
        Série ou lista de séries com os dados. Devem ter índice do tipo 'PeriodIndex'.

    labels: list, default = None
        Lista com labels que serão passados para a função do seaborn.
        Deve ser do mesmo tamanho que a lista de séries.

    figsize: default = plt.figaspect(0.25)
        Parâmetro a ser passado para o matplotlib
    
    Returns
    -------

    fig: plt.Figure
        Figura do matplotlib.
    
    ax: plt.Axis
        Eixo do matplotlib.
    """
    # Cria lista com Nones
    if labels == None:
        labels = [None for _ in series]
    elif len(labels) != len(series):
        raise ValueError("O número de labels deve ser igual ao número de séries.")

    # Define cores
    sns.color_palette("colorblind")

    # Cria eixo no padrão do sktime
    fig, ax = plt.subplots(1, figsize=figsize)
    
    for s, l in zip(series, labels):
        if isinstance(s, pd.DataFrame):
            sns.lineplot(y=s.squeeze(), x=s.index.to_timestamp(), label=l, marker="o", ax=ax)
        else:
            sns.lineplot(y=s, x=s.index.to_timestamp(), label=l, marker="o", ax=ax)

    return fig, ax


def index_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    (Deprecated) Converte coluna de meses do DataFrame para formato reconhecido pelo sktime.

    Parameters
    ----------

    df: pd.DataFrame
        DataFrame original. Precisa ter coluna 'month' com datas no formato mmm/yy.

    Returns
    -------
    
    pd.DataFrame
        DataFrame com datas em formato reconhecido pelo sktime.
    """
    print("Teste")
    warnings.warn("Utilize a função 'index_to_period'.", DeprecationWarning, stacklevel=2)
    df['month'] = df['month'].map(parse_date)
    df['month'] = pd.to_datetime(df['month'], format="%Y-%m")
    return df.set_index('month')


def index_to_period(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte coluna de meses do DataFrame para pd.PeriodIndex

    Parameters
    ----------

    df: pd.DataFrame
        DataFrame original. Precisa ter coluna 'month' com datas no formato mmm/yy.

    Returns
    -------
    
    pd.DataFrame
        DataFrame com datas em formato reconhecido pelo sktime.
    """
    if "/" in df['month'][0]:
        df['month'] = df['month'].map(parse_date)
    df['month'] = pd.to_datetime(df['month'], format="ISO8601")
    df = df.set_index('month')
    df.index = df.index.to_period("M")
    df.index.name = "month"
    return df


def transform_dataframe(df: pd.DataFrame, f, skip=[0]) -> pd.DataFrame:
    """
    Cria novo DataFrame com valores do 'original_dataframe' transformados
    pela função 'f'

    Parameters
    ----------

    df: pd.DataFrame
        DataFrame com os valores originais

    f: Callable
        Função que realiza transformação. Deve ter "signature" f(df, i, col_name)

    skip: list of int, default=[0]
        Lista de linhas a pular. Necessário devido à mudança nos graus de integração

    Returns
    -------
    pd.DataFrame
        Novo DataFrame com valores transformados
    """
    # Cria dicionário de listas vazias com nomes de colunas
    delta_dict = {}
    for col_name, _ in df.items():
        if col_name == "month":
            continue
        # Coloca lixo na primeira linha
        # Vai ser deletado de qualquer forma, então não importa
        delta_dict[col_name] = [0 for _ in range(len(skip))]

    # Popula novo DataFrame com a diferença dos dados no primeiro
    for i, row in df.iterrows():
        # Pula primeira linha
        if i in skip:
            continue
        for col_name, _ in row.items():
            # Ignora coluna de meses
            if col_name == "month":
                continue
            # Salva diferenças em dicionário
            delta_dict[col_name].append(f(df, i, col_name))

    delta_df = pd.DataFrame(delta_dict)

    # Usa coluna de meses como índice
    tmp = df['month']
    delta_df['month'] = tmp
    # Remove as primeiras linhas, pois não contêm valores
    delta_df = delta_df.iloc[skip[-1] + 1:]
    delta_df = delta_df.set_index('month')

    return delta_df


def read_and_change_index(path: str) -> pd.DataFrame:
    """
    Lê arquivo csv e retorna DataFrame com índice mudado por 'index_to_period'.

    Parameters
    ----------
    path: str
        Path para arquivo a ser lido

    Returns
    -------
    pd.DataFrame:
        DataFrame com índice no formato pd.PeriodIndex
    """
    df = pd.read_csv(path, sep=";", decimal=",", thousands=".")
    return index_to_period(df)


def save_tranformed_df(filepath: str, df: pd.DataFrame, f, skip=[0]) -> None:
    """
    Salva arquivo csv com valores de 'df' transformados segundo função 'f'

    Parameters
    ----------
    filepath: str
        Path onde salvar o arquivo novo

    df: pd.DataFrame
        DataFrame com os valores originais

    f: Callable
        Função que realiza transformação. Deve ter "signature" f(df, i, col_name)

    skip: list of int, default=[0]
        Lista de linhas a pular. Necessário devido à mudança nos graus de integração

    Returns
    -------
    None
    """

    delta_df = transform_dataframe(df, f, skip=skip)
    delta_df.to_csv(filepath, sep=";", decimal=",", index_label='month')


def test_for_stationarity_kpss(df: pd.DataFrame) -> dict:
    """
    Realiza teste de estacionariedade KPSS coluna a coluna em 'df'.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame em que se realizará o teste de estacionariedade

    Returns
    -------
    dict:
        Dicionário que contém resultados do teste. Salvo no formato 'coluna': 'resultado'.
        Resultado é instância da classe 'StationarityKPSS'.
    """

    results = {}
    for col in df.columns:
        kpss_tester = StationarityKPSS()
        X = df[col]
        results[col] = kpss_tester.fit(X)
    return results


def print_results(results: dict) -> None:
    """
    Printa resultados de teste de estacionariedade. Presume-se que dicionário seja da forma
    {'coluna': 'resultado'}, em que resultado é instância da classe 'StationaryKPSS'.

    Parameters
    ----------
    results: dict
        Dicionário contendo resultados do teste.
    """
    for k, v in results.items():
        print(k)
        print(f"Stationary: {v.stationary_}")
        print(f"P value: {v.pvalue_}")


def print_stationary_series(results: dict) -> None:
    """
    Printa séries estacionárias. Presume-se que dicionário seja da forma
    {'coluna': 'resultado'}, em que resultado é instância da classe 'StationaryKPSS'.

    Parameters
    ----------
    results: dict
        Dicionário contendo resultados do teste.
    """
    for k, v in results.items():
        if v.stationary_:
            print(k)

