SELECT 
    booking_id,
    no_show,
    branch,
    booking_month,
    arrival_month,
    arrival_day,
    checkout_month,
    checkout_day,
    country,
    first_time,
    room,
    price,
    platform,
    num_adults,
    num_children
FROM noshow 
WHERE no_show IS NOT NULL;
