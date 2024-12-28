# Semantic Search with Vectorized Database
<div align="center">
   <img align="center" height=310px src="https://cdn.dribbble.com/users/24711/screenshots/3886002/media/cf7c84efc880bb82ca058f764833a073.gif" alt="logo">
</div>

## 📝Table of Contents
- [Overview](#Overview)
- [Components](#Components)
- [Installation](#Installation)
- [Usage](#Usage)
- [Contributors](#Contributors)


## 🔍Overview
    This project implements a semantic search engine using a vectorized database. 
    The project aims to build an efficient indexing system to retrieve the top-k most similar vectors based on a given query vector. 



## 🧱Components
-   `VecDB` : A class representing the vectorized database, responsible for storing and retrieving vectors.
- `retrieve()` : A method to retrieve the top-k most similar based on a given query vector.
- `_build_index()` : The function responsible for the indexing.

## 💻Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/Sara-Gamal12/Semantic-Search-with-Vectorized-Database
    ```
2. Navigate to the project directory:
    ```bash
    cd Semantic-Search-with-Vectorized-Database
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the evaluation to get the score and time to retrive the records.
## 🤔Usage
To use this project, follow these steps:

1. Import the `VecDB` class from the module:
    ```python
    from vec_db import VecDB
    ```

2. Initialize the vectorized database:
    ```python
    db = VecDB()
    ```

3. Add vectors to the database:
    ```python
    vectors = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ]
    db.insert_records(vectors)
    ```

4. Retrieve the top-k most similar vectors based on a query vector:
    ```python
    query_vector = [0.1, 0.2, 0.3]
    top_k = 2
    similar_vectors = db.retrieve(query_vector, top_k)
    print(similar_vectors)
    ```

5. Evaluate the performance:
    ```python
    from evaluation import eval, run_queries

    # Example usage
    all_db = db.get_all_rows()
    num_runs = 10  # Define the number of runs
    score, retrieval_time = eval(run_queries(db, all_db, top_k, num_runs))
    print(f"Score: {score}, Retrieval Time: {retrieval_time}")
    ```

## ⭐Contributors
<table  align='center'> 
<tr>
    <td align="center">
        <a href="https://github.com/yousefosama654">
            <img src="https://avatars.githubusercontent.com/u/93356614?v=4" width="100;" alt="yousefosama654"/>
            <br />
            <sub><b>Yousef</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/EmanElbedwihy">
            <img src="https://avatars.githubusercontent.com/u/120182209?v=4" width="100;" alt="EmanElbedwihy"/>
            <br />
            <sub><b>Eman</b></sub>
        </a>
    </td>
        <td align="center">
        <a href="https://github.com/nesma-shafie">
            <img src="https://avatars.githubusercontent.com/u/120175134?v=4" width="100;" alt="nesma-shafie"/>
            <br />
            <sub><b>Nesma</b></sub>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/Sara-Gamal12">
            <img src="https://avatars.githubusercontent.com/u/106556638?v=4" width="100;" alt="Sara-Gamal1"/>
            <br />
            <sub><b>Sara</b></sub>
        </a>
    </td></tr>
</table>
<!-- readme: collaborators -end -->
<h2 align='center'>Thank You. 💖 </h2>
