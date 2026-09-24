"""
Автотесты к ноутбуку «Стандартные контейнеры Python» (python_containers.ipynb).

Файл должен лежать в одной папке с ноутбуком. Использование в ячейке задачи:

    from tests import test_task1

    def task1():
        s = input()
        print(s[::-1])

    test_task1(task1)

Каждая функция test_taskN(func) запускает func на наборе тестов: подменяет
input() строками из теста, перехватывает всё, что печатает print(), и
сравнивает результат с ожидаемым ответом. Отчёт печатается прямо в ячейке.

Правила сравнения:
  * пробелы в конце строк и пустые строки в конце вывода не учитываются;
  * в задачах с вещественным ответом (6 и 14) числа сравниваются
    с точностью FLOAT_TOLERANCE, остальное — посимвольно;
  * если условие запрещает какие-то функции или методы, проверка
    заглядывает в исходный код решения и сообщает о нарушении.

Полезно при отладке — запустить решение на своём входе:

    from tests import run_with_input
    print(run_with_input(task1, "hello"))          # напечатает "olleh"
    print(run_with_input(task6, "3\\n1 2 3"))        # строки входа разделяются \\n
"""

import ast
import builtins
import contextlib
import inspect
import io
import sys
import textwrap
import traceback

__all__ = ["run_with_input", "FLOAT_TOLERANCE"] + [f"test_task{i}" for i in range(1, 36)]

FLOAT_TOLERANCE = 1e-4     # допуск для вещественных ответов
_MAX_SHOWN_LINE = 120      # длиннее этого строки в отчёте обрезаются


# ---------------------------------------------------------------------------
# Запуск решения с подменой ввода/вывода
# ---------------------------------------------------------------------------

def run_with_input(func, stdin_text):
    """Запускает func(), подставляя stdin_text вместо ввода.

    Возвращает всё, что функция напечатала через print(), одной строкой.
    Строки входа разделяются символом перевода строки: "3\\n1 2 3".
    """
    output, _ = _run(func, stdin_text)
    return output


def _run(func, stdin_text):
    lines = iter(stdin_text.split("\n"))

    def fake_input(prompt=""):
        try:
            return next(lines)
        except StopIteration:
            raise EOFError("входные данные закончились") from None

    buffer = io.StringIO()
    saved_input, saved_stdin = builtins.input, sys.stdin
    builtins.input = fake_input
    sys.stdin = io.StringIO(stdin_text + "\n")
    try:
        with contextlib.redirect_stdout(buffer):
            returned = func()
    finally:
        builtins.input, sys.stdin = saved_input, saved_stdin
    return buffer.getvalue(), returned


# ---------------------------------------------------------------------------
# Сравнение вывода с ожидаемым
# ---------------------------------------------------------------------------

def _normalize(text):
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def _tokens_match(got_line, expected_line):
    got, expected = got_line.split(), expected_line.split()
    if len(got) != len(expected):
        return False
    for a, b in zip(got, expected):
        if a == b:
            continue
        try:
            if abs(float(a) - float(b)) <= FLOAT_TOLERANCE:
                continue
        except ValueError:
            pass
        return False
    return True


def _outputs_match(got, expected, floats):
    got_lines, expected_lines = _normalize(got), _normalize(expected)
    if not floats:
        return got_lines == expected_lines
    if len(got_lines) != len(expected_lines):
        return False
    return all(_tokens_match(g, e) for g, e in zip(got_lines, expected_lines))


# ---------------------------------------------------------------------------
# Проверка ограничений вида «не используя sorted()»
# ---------------------------------------------------------------------------

def _find_forbidden(func, names=(), attrs=(), negative_step=False):
    """Ищет в исходном коде функции запрещённые вызовы. Возвращает список описаний."""
    try:
        source = textwrap.dedent(inspect.getsource(func))
        tree = ast.parse(source)
    except Exception:            # исходник недоступен — проверку пропускаем
        return []
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in names:
            found.add(f"{node.id}()")
        elif isinstance(node, ast.Attribute) and node.attr in attrs:
            found.add(f".{node.attr}()")
        elif negative_step and isinstance(node, ast.Slice) and node.step is not None:
            step = node.step
            if (isinstance(step, ast.UnaryOp) and isinstance(step.op, ast.USub)
                    and isinstance(step.operand, ast.Constant)):
                found.add("срез с отрицательным шагом")
    return sorted(found)


# ---------------------------------------------------------------------------
# Отчёт
# ---------------------------------------------------------------------------

def _show(label, text, indent="     "):
    print(f"{indent}{label}:")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines:
        print(f"{indent}  (пусто)")
    for line in lines:
        if len(line) > _MAX_SHOWN_LINE:
            line = line[:_MAX_SHOWN_LINE] + "…"
        print(f"{indent}  {line}")


def _describe_exception(exc, func):
    if isinstance(exc, EOFError):
        return ("функция вызвала input() больше раз, чем есть строк во входных данных "
                "(проверьте формат ввода в условии)")
    where = ""
    for frame in reversed(traceback.extract_tb(exc.__traceback__)):
        if frame.filename == func.__code__.co_filename:
            code = f": {frame.line.strip()}" if frame.line else ""
            where = f" (строка {frame.lineno}{code})"
            break
    return f"{type(exc).__name__}: {exc}{where}"


def _check(number, func):
    task = _TASKS[number]
    if not callable(func):
        print(f"❌ test_task{number}: нужно передать саму функцию, а не {type(func).__name__}. "
              f"Пишите test_task{number}(task{number}) — без скобок после task{number}.")
        return

    cases = task["cases"]
    print(f"Задача {number}. {task['title']} — тестов: {len(cases)}")

    violations = _find_forbidden(func, task.get("forbid_names", ()),
                                 task.get("forbid_attrs", ()), task.get("forbid_negative_step", False))
    if violations:
        print(f"  ⛔ в решении используется запрещённое условием: {', '.join(violations)}")

    passed = 0
    for i, (stdin_text, expected) in enumerate(cases, 1):
        try:
            got, returned = _run(func, stdin_text)
        except Exception as exc:
            print(f"  ❌ тест {i}: ошибка выполнения — {_describe_exception(exc, func)}")
            _show("ввод", stdin_text)
            continue
        if _outputs_match(got, expected, task.get("floats", False)):
            passed += 1
            print(f"  ✅ тест {i}: OK")
            continue
        print(f"  ❌ тест {i}: неверный ответ")
        _show("ввод", stdin_text)
        _show("ожидалось", expected)
        _show("получено", got)
        if returned is not None and got.strip() == "":
            print("     подсказка: функция вернула значение через return, "
                  "а ответ нужно напечатать через print()")

    if passed == len(cases) and not violations:
        print(f"Итог: все {len(cases)} тестов пройдены ✅")
    elif passed == len(cases):
        print(f"Итог: тесты пройдены, но нарушено ограничение условия ❌")
    else:
        print(f"Итог: пройдено {passed} из {len(cases)} ❌")


# ---------------------------------------------------------------------------
# Тесты: {номер: {"title": ..., "cases": [(ввод, ожидаемый вывод), ...], ...}}
# Первые несколько тестов каждой задачи приведены в условии как примеры.
# ---------------------------------------------------------------------------

_TASKS = {
    # ---------------- Базовый уровень: строки ----------------
    1: {
        "title": "Перевёрнутая строка",
        "cases": [
            ("python", "nohtyp"),
            ("hello world", "dlrow olleh"),
            ("a", "a"),
            ("12345", "54321"),
            ("Ab, cd!", "!dc ,bA"),
            ("тест", "тсет"),
            ("racecar", "racecar"),
            ("Python 3.11", "11.3 nohtyP"),
        ],
    },
    2: {
        "title": "Палиндром",
        "cases": [
            ("А роза упала на лапу Азора", "YES"),
            ("hello", "NO"),
            ("Was it a car or a cat I saw", "YES"),
            ("a", "YES"),
            ("ab ba", "YES"),
            ("Python", "NO"),
            ("A man a plan a canal Panama", "YES"),
            ("abc cba x", "NO"),
            ("Кот", "NO"),
        ],
    },
    3: {
        "title": "Подсчёт гласных",
        "cases": [
            ("Hello, World!", "3"),
            ("Привет, мир", "3"),
            ("xyz", "0"),
            ("AEIOU aeiou", "10"),
            ("Ёжик и ёлка", "5"),
            ("Python 3.11", "1"),
            ("Queueing", "5"),
            ("ы", "1"),
        ],
    },
    4: {
        "title": "Пробелы в подчёркивания",
        "cases": [
            ("hello big world", "hello_big_world"),
            ("a  b", "a__b"),
            ("python", "python"),
            ("один два  три   четыре", "один_два__три___четыре"),
            ("x y", "x_y"),
        ],
    },
    5: {
        "title": "Заглавные буквы вручную",
        "forbid_attrs": ("title", "capitalize"),
        "cases": [
            ("hello world", "Hello World"),
            ("python is fun", "Python Is Fun"),
            ("привет мир", "Привет Мир"),
            ("a b c", "A B C"),
            ("x", "X"),
            ("ёж и уж", "Ёж И Уж"),
        ],
    },
    # ---------------- Базовый уровень: списки ----------------
    6: {
        "title": "Сумма и среднее",
        "floats": True,
        "cases": [
            ("5\n1 2 3 4 5", "15 3.0"),
            ("4\n10 20 30 45", "105 26.25"),
            ("1\n-7", "-7 -7.0"),
            ("3\n1 2 2", "5 1.6666666666666667"),
            ("6\n0 0 0 0 0 0", "0 0.0"),
            ("2\n1000000 -1000000", "0 0.0"),
            ("4\n1 1 1 2", "5 1.25"),
        ],
    },
    7: {
        "title": "Без повторов, с сохранением порядка",
        "cases": [
            ("7\n3 1 3 2 1 4 2", "3 1 2 4"),
            ("5\n5 5 5 5 5", "5"),
            ("4\n1 2 3 4", "1 2 3 4"),
            ("6\n2 -1 2 -1 0 2", "2 -1 0"),
            ("8\n4 3 2 1 1 2 3 4", "4 3 2 1"),
        ],
    },
    8: {
        "title": "Второй по величине",
        "forbid_names": ("sorted",),
        "forbid_attrs": ("sort",),
        "cases": [
            ("5\n5 1 9 3 7", "7"),
            ("2\n-3 -10", "-10"),
            ("4\n100 200 300 400", "300"),
            ("3\n1 3 2", "2"),
            ("5\n-1 -2 -3 -4 -5", "-2"),
            ("6\n10 9 8 7 6 5", "9"),
            ("2\n1000000000 -1000000000", "-1000000000"),
        ],
    },
    9: {
        "title": "Разворот списка вручную",
        "forbid_names": ("reversed",),
        "forbid_attrs": ("reverse",),
        "forbid_negative_step": True,
        "cases": [
            ("5\n1 2 3 4 5", "5 4 3 2 1"),
            ("4\n10 20 30 40", "40 30 20 10"),
            ("1\n7", "7"),
            ("6\n1 1 2 2 3 3", "3 3 2 2 1 1"),
            ("2\n-5 5", "5 -5"),
            ("3\n1 2 1", "1 2 1"),
        ],
    },
    10: {
        "title": "Чётные и нечётные",
        "cases": [
            ("6\n1 2 3 4 5 6", "2 4 6\n1 3 5"),
            ("3\n1 3 5", "\n1 3 5"),
            ("4\n0 -2 -3 8", "0 -2 8\n-3"),
            ("2\n4 8", "4 8\n"),
            ("5\n7 -7 0 3 -4", "0 -4\n7 -7 3"),
        ],
    },
    # ---------------- Базовый уровень: кортежи ----------------
    11: {
        "title": "Самый старший",
        "cases": [
            ("3\nАнна 20\nБорис 35\nВера 28", "Борис 35"),
            ("2\nЯн 40\nЕва 40", "Ян 40"),
            ("1\nОлег 7", "Олег 7"),
            ("4\nИван 30\nПётр 31\nМария 31\nОльга 29", "Пётр 31"),
            ("3\na 0\nb 0\nc 0", "a 0"),
        ],
    },
    12: {
        "title": "Обмен местами",
        "cases": [
            ("1 2", "2 1"),
            ("hello world", "world hello"),
            ("x x", "x x"),
            ("left right", "right left"),
        ],
    },
    13: {
        "title": "Сколько раз встречается",
        "cases": [
            ("6\n3 1 3 5 3 2\n3", "3"),
            ("4\n1 2 3 4\n7", "0"),
            ("3\n-1 -1 -1\n-1", "3"),
            ("5\n0 0 1 0 1\n0", "3"),
            ("1\n42\n42", "1"),
        ],
    },
    14: {
        "title": "Длина вектора",
        "floats": True,
        "cases": [
            ("3 4 0", "5.0"),
            ("1 2 2", "3.0"),
            ("1 1 1", "1.7320508075688772"),
            ("0 0 0", "0.0"),
            ("-3 0 4", "5.0"),
            ("2 3 6", "7.0"),
            ("10000 10000 10000", "17320.508075688773"),
        ],
    },
    15: {
        "title": "Объединить и отсортировать",
        "cases": [
            ("3 1 4\n1 5 9", "1 1 3 4 5 9"),
            ("7\n7", "7 7"),
            ("-1 -2\n0 -3", "-3 -2 -1 0"),
            ("5 4 3\n2 1", "1 2 3 4 5"),
            ("1\n1 1 1", "1 1 1 1"),
        ],
    },
    # ---------------- Базовый уровень: словари ----------------
    16: {
        "title": "Частота слов",
        "cases": [
            ("a b a c b a", "a 3\nb 2\nc 1"),
            ("кот пёс кот", "кот 2\nпёс 1"),
            ("x", "x 1"),
            ("to be or not to be", "to 2\nbe 2\nor 1\nnot 1"),
            ("z y x", "z 1\ny 1\nx 1"),
        ],
    },
    17: {
        "title": "Перевёрнутый словарь",
        "cases": [
            ("3\na 1\nb 2\nc 3", "1 a\n2 b\n3 c"),
            ("2\ncat кот\ndog пёс", "кот cat\nпёс dog"),
            ("1\nk v", "v k"),
            ("3\nx y\ny z\nz x", "y x\nz y\nx z"),
        ],
    },
    18: {
        "title": "Самый дорогой товар",
        "cases": [
            ("3\nхлеб 60\nмолоко 90\nсыр 450", "сыр"),
            ("2\nчай 100\nкофе 100", "чай"),
            ("1\nвода 30", "вода"),
            ("4\na 5\nb 7\nc 7\nd 1", "b"),
            ("3\nx 0\ny 0\nz 0", "x"),
        ],
    },
    19: {
        "title": "Слияние словарей с суммированием",
        "cases": [
            ("2\na 1\nb 2\n2\nb 3\nc 4", "a 1\nb 5\nc 4"),
            ("1\nx 10\n1\nx -10", "x 0"),
            ("2\np 1\nq 2\n2\nr 3\nq 4", "p 1\nq 6\nr 3"),
            ("1\na 1\n2\nb 2\nc 3", "a 1\nb 2\nc 3"),
            ("3\nc 3\nb 2\na 1\n3\na 1\nb 2\nc 3", "c 6\nb 4\na 2"),
        ],
    },
    20: {
        "title": "Фильтр по возрасту",
        "cases": [
            ("3\nАня 17\nБоря 22\nВера 19\n18", "Боря 22\nВера 19"),
            ("2\nКим 30\nЛев 25\n30", "Ким 30"),
            ("2\nКим 30\nЛев 25\n31", ""),
            ("3\na 0\nb 1\nc 2\n0", "a 0\nb 1\nc 2"),
            ("4\nw 18\nx 17\ny 18\nz 150\n18", "w 18\ny 18\nz 150"),
        ],
    },
    # ---------------- Базовый уровень: множества ----------------
    21: {
        "title": "Различные элементы",
        "cases": [
            ("6\n1 2 2 3 1 4", "4\n1 2 3 4"),
            ("3\n5 5 5", "1\n5"),
            ("4\n-1 3 -1 2", "3\n-1 2 3"),
            ("1\n0", "1\n0"),
            ("7\n3 3 2 2 1 1 0", "4\n0 1 2 3"),
        ],
    },
    22: {
        "title": "Пересечение списков",
        "cases": [
            ("4\n1 2 3 4\n3\n3 4 5", "3 4"),
            ("3\n1 1 2\n3\n2 2 1", "1 2"),
            ("2\n1 2\n2\n3 4", ""),
            ("5\n5 4 3 2 1\n5\n1 2 3 4 5", "1 2 3 4 5"),
            ("3\n-1 0 1\n2\n0 -1", "-1 0"),
        ],
    },
    23: {
        "title": "Подмножество",
        "cases": [
            ("2\n1 2\n3\n1 2 3", "YES"),
            ("3\n1 2 3\n2\n1 2", "NO"),
            ("2\n4 4\n1\n4", "YES"),
            ("1\n0\n3\n1 2 3", "NO"),
            ("3\n3 2 1\n3\n1 2 3", "YES"),
            ("2\n1 5\n4\n1 2 3 4", "NO"),
        ],
    },
    24: {
        "title": "Разность списков",
        "cases": [
            ("4\n1 2 3 4\n3\n3 4 5", "1 2"),
            ("3\n7 7 8\n1\n9", "7 8"),
            ("2\n1 2\n3\n2 1 0", ""),
            ("5\n5 4 3 2 1\n2\n3 3", "1 2 4 5"),
            ("3\n-2 -1 0\n1\n-1", "-2 0"),
        ],
    },
    25: {
        "title": "Повторяющиеся символы",
        "cases": [
            ("hello", "YES"),
            ("world", "NO"),
            ("aA", "NO"),
            ("abcabc", "YES"),
            ("1234567890", "NO"),
            ("x", "NO"),
            ("aa", "YES"),
        ],
    },
    # ---------------- Средний уровень ----------------
    26: {
        "title": "Три самых частых слова",
        "cases": [
            ("Python is great. Python is easy. I love Python and I love easy code.",
             "python 3\neasy 2\ni 2"),
            ("Мама мыла раму, мама мыла Раму!", "мама 2\nмыла 2\nраму 2"),
            ("Hello, hello!!! HELLO?", "hello 3"),
            ("one two two three three three", "three 3\ntwo 2\none 1"),
            ("a b", "a 1\nb 1"),
            ("Wow... wow: WOW, and yes; yes", "wow 3\nyes 2\nand 1"),
            ("ab12ab 3cd", "ab 2\ncd 1"),
            ("It is what it is, isn't it?", "it 3\nis 2\nisn 1"),
        ],
    },
    27: {
        "title": "Две группы студентов",
        "cases": [
            ("4\nАня Боря Вера Аня\n3\nВера Гриша Боря", "Боря Вера\nАня Гриша"),
            ("2\nAnn Bob\n2\nBob Ann", "Ann Bob\n"),
            ("1\nZed\n1\nAmy", "\nAmy Zed"),
            ("3\na b c\n3\nb c d", "b c\na d"),
            ("2\nx x\n1\nx", "x\n"),
        ],
    },
    28: {
        "title": "Стоимость по товарам",
        "cases": [
            ("4\nхлеб 60 3\nмолоко 90 2\nхлеб 60 1\nсыр 450 1", "сыр 450\nхлеб 240\nмолоко 180"),
            ("3\na 10 1\nb 5 2\nc 1 10", "a 10\nb 10\nc 10"),
            ("2\npen 3 4\npen 1 1", "pen 13"),
            ("3\nbook 100 1\npen 10 5\ncup 25 2", "book 100\ncup 50\npen 50"),
            ("1\nx 1000000 1000000", "x 1000000000000"),
        ],
    },
    29: {
        "title": "Длинные слова",
        "cases": [
            ("Python is a powerful, flexible and popular programming language.",
             "flexible language popular powerful programming python"),
            ("Кто ходит в гости по утрам, тот поступает мудро!", "гости мудро поступает утрам ходит"),
            ("Hi all, how are you?", ""),
            ("Apple apple APPLE!", "apple"),
            ("Short words only here", "short words"),
            ("exactly four1chars", "chars exactly"),
        ],
    },
    30: {
        "title": "Группировка по ключу",
        "cases": [
            ("4\nfruits apple\nveg carrot\nfruits banana\nveg potato",
             "fruits apple banana\nveg carrot potato"),
            ("3\na 1\na 2\na 3", "a 1 2 3"),
            ("2\nx y\nz y", "x y\nz y"),
            ("5\nb 1\na 2\nb 3\nc 4\na 5", "b 1 3\na 2 5\nc 4"),
        ],
    },
    # ---------------- Сложный уровень ----------------
    31: {
        "title": "Инвертированный индекс",
        "cases": [
            ("3\npython is great\njava is verbose\npython is easy and great",
             "and 3\neasy 3\ngreat 1 3\nis 1 2 3\njava 2\npython 1 3\nverbose 2"),
            ("2\na a a\na b", "a 1 2\nb 2"),
            ("1\nx", "x 1"),
            ("4\nred\nblue\nred blue\ngreen", "blue 2 3\ngreen 4\nred 1 3"),
        ],
    },
    32: {
        "title": "Анаграммы",
        "cases": [
            ("Listen\nSilent", "YES"),
            ("Hello\nWorld", "NO"),
            ("Dormitory\nDirty room!!", "YES"),
            ("The eyes\nThey see", "YES"),
            ("aab\nabb", "NO"),
            ("A gentleman\nElegant man", "YES"),
            ("abc\nabcd", "NO"),
            ("Апельсин\nСпаниель", "YES"),
        ],
    },
    33: {
        "title": "Дедупликация записей",
        "cases": [
            ("id name city\nid\n4\n1 Аня Москва\n2 Боря Казань\n1 Аня Сочи\n3 Вера Тверь",
             "1 Аня Москва\n2 Боря Казань\n3 Вера Тверь"),
            ("id name city\ncity\n4\n1 Аня Москва\n2 Боря Казань\n3 Вера Москва\n4 Гоша Казань",
             "1 Аня Москва\n2 Боря Казань"),
            ("login\nlogin\n3\nadmin\nadmin\nroot", "admin\nroot"),
            ("a b\nb\n3\n1 x\n2 y\n3 x", "1 x\n2 y"),
        ],
    },
    34: {
        "title": "Соседи за два шага",
        "cases": [
            ("4\nA B\nB C\nC D\nA E\nA", "B C E"),
            ("4\nA B\nB C\nC D\nA E\nD", "B C"),
            ("2\nA B\nC D\nX", ""),
            ("3\na b\nb c\nc a\na", "b c"),
            ("1\np q\nq", "p"),
            ("5\nA B\nB C\nC D\nD E\nE F\nC", "A B D E"),
        ],
    },
    35: {
        "title": "Шифр Цезаря",
        "cases": [
            ("encrypt\n3\nHello, World!", "Khoor, Zruog!"),
            ("decrypt\n3\nKhoor, Zruog!", "Hello, World!"),
            ("crack\n"
             "Wfaovu pz h nlulyhs wbywvzl wyvnyhttpun shunbhnl aoha slaz fvb dvyr xbpjrsf huk "
             "pualnyhal zfzaltz tvyl lmmljapclsf. Pa ltwohzpglz jvkl ylhkhipspaf dpao aol bzl vm "
             "zpnupmpjhua pukluahapvu.",
             "7\n"
             "Python is a general purpose programming language that lets you work quickly and "
             "integrate systems more effectively. It emphasizes code readability with the use of "
             "significant indentation."),
            ("encrypt\n29\nabc xyz", "def abc"),
            ("encrypt\n-1\nAbc", "Zab"),
            ("decrypt\n0\nsame text 123", "same text 123"),
            ("decrypt\n-2\nabc", "cde"),
            ("crack\n"
             "Gur dhvpx oebja sbk whzcf bire gur ynml qbt. Guvf fragrapr pbagnvaf rirel yrggre bs "
             "gur Ratyvfu nycunorg, juvpu znxrf vg hfrshy sbe grfgvat sbagf naq xrlobneqf.",
             "13\n"
             "The quick brown fox jumps over the lazy dog. This sentence contains every letter of "
             "the English alphabet, which makes it useful for testing fonts and keyboards."),
            ("crack\n"
             "It was the best of times, it was the worst of times, it was the age of wisdom, it was "
             "the age of foolishness, it was the epoch of belief, it was the epoch of incredulity.",
             "0\n"
             "It was the best of times, it was the worst of times, it was the age of wisdom, it was "
             "the age of foolishness, it was the epoch of belief, it was the epoch of incredulity."),
            ("crack\n"
             "Lzmx cdudknodqr zqntmc sgd vnqkc trd oxsgnm dudqx rhmfkd czx enq czsz rbhdmbd, vda "
             "cdudknoldms, ztsnlzshnm zmc zqshehbhzk hmsdkkhfdmbd qdrdzqbg.",
             "25\n"
             "Many developers around the world use python every single day for data science, web "
             "development, automation and artificial intelligence research."),
        ],
    },
}


# ---------------------------------------------------------------------------
# Публичные функции проверки: test_taskN(taskN)
# ---------------------------------------------------------------------------

def test_task1(func):
    """Задача 1. Перевёрнутая строка."""
    _check(1, func)


def test_task2(func):
    """Задача 2. Палиндром."""
    _check(2, func)


def test_task3(func):
    """Задача 3. Подсчёт гласных."""
    _check(3, func)


def test_task4(func):
    """Задача 4. Пробелы в подчёркивания."""
    _check(4, func)


def test_task5(func):
    """Задача 5. Заглавные буквы вручную."""
    _check(5, func)


def test_task6(func):
    """Задача 6. Сумма и среднее."""
    _check(6, func)


def test_task7(func):
    """Задача 7. Без повторов, с сохранением порядка."""
    _check(7, func)


def test_task8(func):
    """Задача 8. Второй по величине."""
    _check(8, func)


def test_task9(func):
    """Задача 9. Разворот списка вручную."""
    _check(9, func)


def test_task10(func):
    """Задача 10. Чётные и нечётные."""
    _check(10, func)


def test_task11(func):
    """Задача 11. Самый старший."""
    _check(11, func)


def test_task12(func):
    """Задача 12. Обмен местами."""
    _check(12, func)


def test_task13(func):
    """Задача 13. Сколько раз встречается."""
    _check(13, func)


def test_task14(func):
    """Задача 14. Длина вектора."""
    _check(14, func)


def test_task15(func):
    """Задача 15. Объединить и отсортировать."""
    _check(15, func)


def test_task16(func):
    """Задача 16. Частота слов."""
    _check(16, func)


def test_task17(func):
    """Задача 17. Перевёрнутый словарь."""
    _check(17, func)


def test_task18(func):
    """Задача 18. Самый дорогой товар."""
    _check(18, func)


def test_task19(func):
    """Задача 19. Слияние словарей с суммированием."""
    _check(19, func)


def test_task20(func):
    """Задача 20. Фильтр по возрасту."""
    _check(20, func)


def test_task21(func):
    """Задача 21. Различные элементы."""
    _check(21, func)


def test_task22(func):
    """Задача 22. Пересечение списков."""
    _check(22, func)


def test_task23(func):
    """Задача 23. Подмножество."""
    _check(23, func)


def test_task24(func):
    """Задача 24. Разность списков."""
    _check(24, func)


def test_task25(func):
    """Задача 25. Повторяющиеся символы."""
    _check(25, func)


def test_task26(func):
    """Задача 26. Три самых частых слова."""
    _check(26, func)


def test_task27(func):
    """Задача 27. Две группы студентов."""
    _check(27, func)


def test_task28(func):
    """Задача 28. Стоимость по товарам."""
    _check(28, func)


def test_task29(func):
    """Задача 29. Длинные слова."""
    _check(29, func)


def test_task30(func):
    """Задача 30. Группировка по ключу."""
    _check(30, func)


def test_task31(func):
    """Задача 31. Инвертированный индекс."""
    _check(31, func)


def test_task32(func):
    """Задача 32. Анаграммы."""
    _check(32, func)


def test_task33(func):
    """Задача 33. Дедупликация записей."""
    _check(33, func)


def test_task34(func):
    """Задача 34. Соседи за два шага."""
    _check(34, func)


def test_task35(func):
    """Задача 35. Шифр Цезаря."""
    _check(35, func)
