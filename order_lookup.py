import json


def load_orders():
    with open("assignment-data/data/orders.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["orders"]


def lookup_order(order_id):
    """Looks up an order and returns ONLY customer-safe fields."""
    order_id = order_id.strip().upper()  # clean up messy input (spaces, lowercase)

    orders = load_orders()
    for order in orders:
        if order["order_id"] == order_id:
            return {
                "found": True,
                "order_id": order["order_id"],
                "status": order["status"],
                "items": [item["name"] for item in order["items"]],
                "carrier": order.get("carrier"),
                "tracking_number": order.get("tracking_number"),
                "estimated_delivery": order.get("estimated_delivery"),
                "message": order.get("customer_safe_message"),
            }

    return {"found": False, "order_id": order_id}


if __name__ == "__main__":
    print(lookup_order("ORD-1001"))
    print(lookup_order("ord-1002"))     # test messy lowercase input
    print(lookup_order("ORD-9999"))     # test an ID that doesn't exist