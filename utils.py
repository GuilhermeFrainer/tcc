import pandas as pd
import warnings
from sktime.param_est.stationarity import StationarityKPSS


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

    month, year = s.split("/")
    month = month_dict[month]

    return f"20{year}-{month:02d}-01"


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

