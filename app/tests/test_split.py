"""Stub test for app_cell_split.py (split the bill): builds the app with the smoke-test stubs, checks the split maths,
the group ledger and settle-up, the error paths, and the API wiring over HTTP. Run: .venv/Scripts/python.exe -I test_split.py"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "smoke_test.py"), encoding="utf-8").read().split("# ---------- A.")[0]
H = {"__file__": os.path.join(HERE, "smoke_test.py"), "__name__": "harness"}
exec(src, H)
H["app_cells"] = lambda: [open(os.path.join(HERE, "app_cell_split.py"), encoding="utf-8").read(), ""]
check, pd, gr = H["check"], H["pd"], H["gr"]

ns, m = H["namespace"]()
cs, add, settle, rows, fill = ns["calc_split"], ns["add_to_group"], ns["_settle"], ns["_rows"], ns["fill_split_items"]

# equal split: cents remainder goes to the payer, shares add up exactly
html, st = cs("Me, Riya, Arjun", "me", "Equally", None, "193.00", "OJC MARKETING")
check("equal: shares sum to the total", abs(sum(st["shares"].values()) - 193.00) < 1e-9)
check("equal: payer matched case-insensitively and gets the extra cent", st["payer"] == "Me" and st["shares"]["Me"] == 64.34)
check("equal: html says who owes whom", "owes Me" in html and "Left to split" in html)

# by items with Others (tax/service) shared in proportion, unknown name falls back to everyone
items = pd.DataFrame([["Nasi", "30.00", "Riya"], ["Teh", "RM 9.00", "Arjun, Me"], ["Ayam", "40.00", "all"], ["Kopi", "4.00", "Bob"]],
                     columns=["item", "price", "shared by"])
html, st = cs("Me, Riya, Arjun", "Riya", "By items", items, "89.00", "WARUNG")
check("by items: shares sum to the total", abs(sum(st["shares"].values()) - 89.00) < 1e-9)
check("by items: Riya pays more than Arjun (she had the rice)", st["shares"]["Riya"] > st["shares"]["Arjun"])
check("by items: Others row shown", "Others (tax, service, rounding)" in html and "6.00" in html)
check("by items: unknown name noted", "nobody called" in html)

# big Others triggers the 'make sure' warning (prices may be unit prices)
html, _ = cs("A, B", "A", "By items", pd.DataFrame([["x", "10", "all"]], columns=["item", "price", "shared by"]), "50", "")
check("big Others -> warning", "Make sure this amount is correct" in html)

# discount (negative Others) still sums to the total
html, st = cs("A, B", "A", "By items", pd.DataFrame([["x", "60", "A"], ["y", "40", "B"]], columns=["item", "price", "shared by"]), "90", "")
check("discount: shares sum to the total and stay proportional", abs(sum(st["shares"].values()) - 90) < 1e-9 and st["shares"]["A"] == 54.0)

# API-style dict input
check("dict rows from the API", rows({"headers": ["item", "price", "shared by"], "data": [["x", "1", "all"]]}) == [["x", "1", "all"]])

# group ledger + fewest-payments settle-up
ns["GROUP"]["balances"].clear(); ns["GROUP"]["history"].clear()
_, st1 = cs("Me, Riya, Arjun", "Me", "Equally", None, "90", "Lunch")
_, st2 = cs("Me, Riya, Arjun", "Riya", "Equally", None, "30", "Coffee")
add(st1); g_html, path = add(st2)
bal = ns["GROUP"]["balances"]
check("ledger: balances sum to zero", abs(sum(bal.values())) < 1e-9)
check("ledger: Me +50, Riya -10, Arjun -40 (lunch 90 paid by Me, coffee 30 paid by Riya)",
      abs(bal["Me"] - 50) < 1e-9 and abs(bal["Riya"] + 10) < 1e-9 and abs(bal["Arjun"] + 40) < 1e-9)
mv = settle(dict(bal))
check("settle: Arjun pays Me 40, Riya pays Me 10", mv == [("Arjun", "Me", 40.0), ("Riya", "Me", 10.0)])
check("ledger: CSV written", path and os.path.exists(path))
check("ledger html", "Settle up" in g_html and "Arjun" in g_html)

# fill items: from the line-items table, else via extract_line_items
df, radio = fill(None, pd.DataFrame([["NASI", "1", "30.00"]], columns=["name", "quantity", "price"]))
check("fill from line-items table", df.values.tolist() == [["NASI", "30.00", "all"]])
df, radio = fill(H["receipt"]("X"), pd.DataFrame(columns=["name", "quantity", "price"]))
check("fill via extract_line_items when the table is empty", len(df) == 2 and df.iloc[0, 2] == "all")

# error paths
for args, name in (((" Me ", "Me", "Equally", None, "10", ""), "one person"), (("A, B", "C", "Equally", None, "10", ""), "payer not listed"),
                   (("A, B", "A", "Equally", None, "", ""), "no total"), (("A, B", "A", "By items", None, "10", ""), "no items")):
    try:
        cs(*args); check(f"error: {name}", False)
    except gr.Error:
        check(f"error: {name}", True)

# HTTP: launch locally and call the new endpoints through gradio_client
from gradio_client import Client
demo = ns["demo"]
demo.launch(prevent_thread_lock=True, server_port=7866, theme=ns["APP_THEME"], css=ns["APP_CSS"], quiet=True)
try:
    c = Client("http://127.0.0.1:7866/", verbose=False)
    r = c.predict("Me, Riya", "Me", "By items", {"headers": ["item", "price", "shared by"], "data": [["A", "10", "Riya"], ["B", "10", "all"]]},
                  "22.00", "TEST", api_name="/calc_split")
    check("HTTP /calc_split returns html", "Riya" in str(r) and "owes Me" in str(r))
    r2 = c.predict(api_name="/add_to_group")
    check("HTTP /add_to_group returns group html", "Group balances" in str(r2))
finally:
    demo.close()
fails = [n for n, ok in H["checks"] if not ok]
print(f"\n{len(H['checks']) - len(fails)}/{len(H['checks'])} passed", "| FAILED:", fails if fails else "none")
sys.exit(1 if fails else 0)
