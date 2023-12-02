import pandas as pd


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

