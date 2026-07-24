-- ════════════════════════════════════════════════════════════════
-- KLYPO AFILIADOS — FASE 1 COMPLETO
-- Ejecutar de una sola vez en Supabase SQL Editor (rol postgres)
-- ════════════════════════════════════════════════════════════════


-- ────────────────────────────────────────────────────────────────
-- 1. TABLAS
-- ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS packs (
    id             TEXT          PRIMARY KEY,
    precio_usd     NUMERIC(10,2) NOT NULL CHECK (precio_usd > 0),
    comision_pct   NUMERIC(5,4)  NOT NULL CHECK (comision_pct > 0 AND comision_pct < 1),
    activo         BOOLEAN       NOT NULL DEFAULT true,
    actualizado_en TIMESTAMPTZ   NOT NULL DEFAULT now()
);

INSERT INTO packs (id, precio_usd, comision_pct) VALUES
    ('starter',  5.99, 0.15),
    ('pro',     11.99, 0.15),
    ('creator', 22.99, 0.15)
ON CONFLICT (id) DO NOTHING;


CREATE TABLE IF NOT EXISTS afiliados (
    id        UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id   UUID          NOT NULL UNIQUE REFERENCES auth.users(id) ON DELETE CASCADE,
    codigo    TEXT          NOT NULL UNIQUE,
    estado    TEXT          NOT NULL DEFAULT 'activo'
                            CHECK (estado IN ('activo','suspendido')),
    saldo     NUMERIC(10,2) NOT NULL DEFAULT 0 CHECK (saldo >= 0),
    creado_en TIMESTAMPTZ   NOT NULL DEFAULT now()
);


CREATE TABLE IF NOT EXISTS referidos (
    id              UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    afiliado_id     UUID          NOT NULL REFERENCES afiliados(id) ON DELETE CASCADE,
    -- buyer_user_id y comprador_email los pone la funcion internamente, nunca el cliente
    buyer_user_id   UUID          NOT NULL,
    comprador_email TEXT          NOT NULL,
    plan            TEXT          NOT NULL,
    monto_pagado    NUMERIC(10,2) NOT NULL,
    comision        NUMERIC(10,2) NOT NULL,
    estado          TEXT          NOT NULL DEFAULT 'pendiente'
                                  CHECK (estado IN ('pendiente','aprobado','rechazado')),
    creado_en       TIMESTAMPTZ   NOT NULL DEFAULT now()
);


CREATE TABLE IF NOT EXISTS retiros (
    id          UUID          PRIMARY KEY DEFAULT gen_random_uuid(),
    afiliado_id UUID          NOT NULL REFERENCES afiliados(id) ON DELETE CASCADE,
    monto       NUMERIC(10,2) NOT NULL CHECK (monto >= 10),
    -- whitelist y limite de longitud a nivel de tabla (defensa en profundidad)
    metodo      TEXT          NOT NULL CHECK (metodo IN ('paypal','transferencia','wise','crypto')),
    datos_pago  TEXT          NOT NULL CHECK (length(datos_pago) BETWEEN 1 AND 200),
    estado      TEXT          NOT NULL DEFAULT 'pendiente'
                              CHECK (estado IN ('pendiente','pagado','rechazado')),
    creado_en   TIMESTAMPTZ   NOT NULL DEFAULT now()
);


CREATE INDEX IF NOT EXISTS idx_afiliados_user_id  ON afiliados(user_id);
CREATE INDEX IF NOT EXISTS idx_afiliados_codigo   ON afiliados(codigo);
CREATE INDEX IF NOT EXISTS idx_referidos_afiliado ON referidos(afiliado_id);
CREATE INDEX IF NOT EXISTS idx_referidos_buyer    ON referidos(buyer_user_id);
CREATE INDEX IF NOT EXISTS idx_retiros_afiliado   ON retiros(afiliado_id);


-- ────────────────────────────────────────────────────────────────
-- 2. RLS — solo SELECT para usuarios; los writes los hacen las funciones
-- ────────────────────────────────────────────────────────────────

ALTER TABLE packs     ENABLE ROW LEVEL SECURITY;
ALTER TABLE afiliados ENABLE ROW LEVEL SECURITY;
ALTER TABLE referidos ENABLE ROW LEVEL SECURITY;
ALTER TABLE retiros   ENABLE ROW LEVEL SECURITY;

-- Precios: lectura publica (el frontend los usa para mostrar planes)
CREATE POLICY "packs_select_publico" ON packs
    FOR SELECT TO anon, authenticated
    USING (activo = true);

-- Afiliado ve solo su propia fila
CREATE POLICY "afiliados_select_propio" ON afiliados
    FOR SELECT TO authenticated
    USING (user_id = auth.uid());

-- Afiliado ve solo sus referidos
CREATE POLICY "referidos_select_propios" ON referidos
    FOR SELECT TO authenticated
    USING (afiliado_id = (SELECT id FROM afiliados WHERE user_id = auth.uid()));

-- Afiliado ve solo sus retiros
CREATE POLICY "retiros_select_propios" ON retiros
    FOR SELECT TO authenticated
    USING (afiliado_id = (SELECT id FROM afiliados WHERE user_id = auth.uid()));


-- ────────────────────────────────────────────────────────────────
-- 3. registrar_afiliado()
--    Quien: usuario logueado que quiere ser afiliado
--    GRANT: authenticated | REVOKE: anon
-- ────────────────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION registrar_afiliado()
RETURNS TEXT
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE
    v_user_id  UUID;
    v_codigo   TEXT;
    v_intentos INT := 0;
BEGIN
    v_user_id := auth.uid();
    IF v_user_id IS NULL THEN RAISE EXCEPTION 'NO_AUTH'; END IF;

    -- Si ya es afiliado, devolver codigo existente (idempotente)
    SELECT codigo INTO v_codigo FROM afiliados WHERE user_id = v_user_id;
    IF FOUND THEN RETURN v_codigo; END IF;

    -- Generar codigo unico: KLYPO + 8 chars hex
    LOOP
        v_codigo := 'KLYPO' || upper(left(replace(gen_random_uuid()::text, '-', ''), 8));
        EXIT WHEN NOT EXISTS (SELECT 1 FROM afiliados WHERE codigo = v_codigo);
        v_intentos := v_intentos + 1;
        IF v_intentos > 10 THEN RAISE EXCEPTION 'CODIGO_COLISION'; END IF;
    END LOOP;

    INSERT INTO afiliados (user_id, codigo) VALUES (v_user_id, v_codigo);
    RETURN v_codigo;
END;
$$;

REVOKE EXECUTE ON FUNCTION registrar_afiliado() FROM PUBLIC, anon, authenticated;
GRANT  EXECUTE ON FUNCTION registrar_afiliado() TO authenticated;


-- ────────────────────────────────────────────────────────────────
-- 4. registrar_referido(codigo, pack)
--    Quien: comprador autenticado despues de pagar
--    buyer_user_id y comprador_email se sacan de auth internamente
--    GRANT: authenticated | REVOKE: anon
-- ────────────────────────────────────────────────────────────────

-- DROP si existe version anterior con firma distinta (6 parametros)
DROP FUNCTION IF EXISTS registrar_referido(TEXT, TEXT, NUMERIC, NUMERIC, TEXT, TEXT);

CREATE OR REPLACE FUNCTION registrar_referido(
    p_codigo_afiliado TEXT,
    p_pack            TEXT    -- 'starter' | 'pro' | 'creator'
)
RETURNS UUID
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE
    v_buyer_id     UUID;
    v_email        TEXT;
    v_afiliado_id  UUID;
    v_afiliado_uid UUID;
    v_precio       NUMERIC;
    v_comision_pct NUMERIC;
    v_comision     NUMERIC;
    v_ref_id       UUID;
BEGIN
    -- Comprador = siempre auth.uid(), nunca viene por parametro
    v_buyer_id := auth.uid();
    IF v_buyer_id IS NULL THEN RAISE EXCEPTION 'NO_AUTH'; END IF;

    -- Email sacado de auth.users internamente (SECURITY DEFINER tiene acceso)
    -- Nunca viene por parametro: el cliente no puede manipularlo
    SELECT email INTO v_email FROM auth.users WHERE id = v_buyer_id;
    IF v_email IS NULL THEN RAISE EXCEPTION 'COMPRADOR_SIN_EMAIL'; END IF;

    -- Buscar afiliado activo
    SELECT id, user_id INTO v_afiliado_id, v_afiliado_uid
    FROM afiliados
    WHERE codigo = upper(trim(p_codigo_afiliado)) AND estado = 'activo';
    IF NOT FOUND THEN RAISE EXCEPTION 'CODIGO_INVALIDO'; END IF;

    -- Anti-auto-referido: funciona de verdad porque buyer viene de auth.uid()
    IF v_afiliado_uid = v_buyer_id THEN RAISE EXCEPTION 'AUTO_REFERIDO_BLOQUEADO'; END IF;

    -- Precio y comision del servidor, nunca del cliente
    SELECT precio_usd, comision_pct INTO v_precio, v_comision_pct
    FROM packs WHERE id = lower(trim(p_pack)) AND activo = true;
    IF NOT FOUND THEN RAISE EXCEPTION 'PACK_INVALIDO'; END IF;

    -- Anti-flood: bloquear solo si hay un PENDIENTE activo para este par buyer+afiliado
    -- Permite nuevas compras si el anterior fue aprobado (compra repetida legitima)
    -- Permite nuevas compras si el anterior fue rechazado (falso rechazado o error admin)
    IF EXISTS (
        SELECT 1 FROM referidos
        WHERE afiliado_id   = v_afiliado_id
          AND buyer_user_id = v_buyer_id
          AND estado        = 'pendiente'
    ) THEN RAISE EXCEPTION 'REFERIDO_PENDIENTE_YA_EXISTE'; END IF;

    v_comision := round(v_precio * v_comision_pct, 2);

    INSERT INTO referidos
        (afiliado_id, buyer_user_id, comprador_email, plan, monto_pagado, comision)
    VALUES
        (v_afiliado_id, v_buyer_id, v_email, lower(trim(p_pack)), v_precio, v_comision)
    RETURNING id INTO v_ref_id;

    RETURN v_ref_id;
END;
$$;

REVOKE EXECUTE ON FUNCTION registrar_referido(TEXT, TEXT) FROM PUBLIC, anon, authenticated;
GRANT  EXECUTE ON FUNCTION registrar_referido(TEXT, TEXT) TO authenticated;


-- ────────────────────────────────────────────────────────────────
-- 5. solicitar_retiro(monto, metodo, datos_pago)
--    Quien: afiliado que quiere cobrar su saldo
--    GRANT: authenticated | REVOKE: anon
-- ────────────────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION solicitar_retiro(
    p_monto      NUMERIC,
    p_metodo     TEXT,
    p_datos_pago TEXT
)
RETURNS UUID
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE
    v_afiliado_id UUID;
    v_rows        INT;
    v_retiro_id   UUID;
BEGIN
    IF auth.uid() IS NULL THEN RAISE EXCEPTION 'NO_AUTH'; END IF;
    IF p_monto < 10 THEN RAISE EXCEPTION 'MONTO_MINIMO_10'; END IF;

    -- Whitelist de metodos (refuerzo en funcion; el CHECK de la tabla es la ultima barrera)
    IF p_metodo NOT IN ('paypal','transferencia','wise','crypto')
        THEN RAISE EXCEPTION 'METODO_INVALIDO'; END IF;

    -- Validacion de datos_pago (refuerzo en funcion)
    IF length(trim(p_datos_pago)) = 0 THEN RAISE EXCEPTION 'DATOS_PAGO_REQUERIDO'; END IF;
    IF length(p_datos_pago) > 200     THEN RAISE EXCEPTION 'DATOS_PAGO_DEMASIADO_LARGOS'; END IF;

    -- UPDATE atomico: verifica saldo Y descuenta en una sola operacion
    -- Elimina la race condition TOCTOU de SELECT-luego-UPDATE
    UPDATE afiliados
    SET    saldo = saldo - p_monto
    WHERE  user_id = auth.uid()
      AND  estado  = 'activo'
      AND  saldo  >= p_monto;

    GET DIAGNOSTICS v_rows = ROW_COUNT;

    IF v_rows = 0 THEN
        IF NOT EXISTS (
            SELECT 1 FROM afiliados WHERE user_id = auth.uid() AND estado = 'activo'
        ) THEN RAISE EXCEPTION 'NO_ES_AFILIADO'; END IF;
        RAISE EXCEPTION 'SALDO_INSUFICIENTE';
    END IF;

    SELECT id INTO v_afiliado_id FROM afiliados WHERE user_id = auth.uid();

    INSERT INTO retiros (afiliado_id, monto, metodo, datos_pago)
    VALUES (v_afiliado_id, p_monto, p_metodo, p_datos_pago)
    RETURNING id INTO v_retiro_id;

    RETURN v_retiro_id;
END;
$$;

REVOKE EXECUTE ON FUNCTION solicitar_retiro(NUMERIC, TEXT, TEXT) FROM PUBLIC, anon, authenticated;
GRANT  EXECUTE ON FUNCTION solicitar_retiro(NUMERIC, TEXT, TEXT) TO authenticated;


-- ────────────────────────────────────────────────────────────────
-- 6. aprobar_comision(UUID) — solo admin / SQL Editor
--    Sin GRANT a nadie. Solo postgres puede ejecutarla.
-- ────────────────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION aprobar_comision(p_referido_id UUID)
RETURNS VOID
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE
    v_afiliado_id UUID;
    v_comision    NUMERIC;
BEGIN
    SELECT afiliado_id, comision INTO v_afiliado_id, v_comision
    FROM referidos WHERE id = p_referido_id AND estado = 'pendiente';
    IF NOT FOUND THEN RAISE EXCEPTION 'REFERIDO_NO_ENCONTRADO_O_YA_PROCESADO'; END IF;

    UPDATE referidos SET estado = 'aprobado'         WHERE id  = p_referido_id;
    UPDATE afiliados SET saldo  = saldo + v_comision  WHERE id  = v_afiliado_id;
END;
$$;

REVOKE EXECUTE ON FUNCTION aprobar_comision(UUID) FROM PUBLIC, anon, authenticated;
-- Sin GRANT adicional. Solo postgres via SQL Editor.


-- ────────────────────────────────────────────────────────────────
-- 7. procesar_retiro(UUID, BOOLEAN) — solo admin / SQL Editor
--    true = pagar | false = rechazar y restaurar saldo
--    Sin GRANT a nadie. Solo postgres puede ejecutarla.
-- ────────────────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION procesar_retiro(p_retiro_id UUID, p_aprobado BOOLEAN)
RETURNS VOID
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE
    v_afiliado_id UUID;
    v_monto       NUMERIC;
BEGIN
    SELECT afiliado_id, monto INTO v_afiliado_id, v_monto
    FROM retiros WHERE id = p_retiro_id AND estado = 'pendiente';
    IF NOT FOUND THEN RAISE EXCEPTION 'RETIRO_NO_ENCONTRADO_O_YA_PROCESADO'; END IF;

    IF p_aprobado THEN
        -- Saldo ya fue descontado en solicitar_retiro; solo marcar pagado
        UPDATE retiros SET estado = 'pagado' WHERE id = p_retiro_id;
    ELSE
        -- Rechazado: restaurar saldo al afiliado
        UPDATE retiros  SET estado = 'rechazado'      WHERE id = p_retiro_id;
        UPDATE afiliados SET saldo = saldo + v_monto  WHERE id = v_afiliado_id;
    END IF;
END;
$$;

REVOKE EXECUTE ON FUNCTION procesar_retiro(UUID, BOOLEAN) FROM PUBLIC, anon, authenticated;
-- Sin GRANT adicional. Solo postgres via SQL Editor.


-- ────────────────────────────────────────────────────────────────
-- 8. VISTA ADMIN — solo accesible desde el SQL Editor
--    Une email del comprador, datos del pack y del afiliado
-- ────────────────────────────────────────────────────────────────

CREATE OR REPLACE VIEW v_referidos_admin AS
SELECT
    r.id                AS referido_id,
    r.creado_en         AS fecha,
    r.comprador_email,
    r.plan,
    r.monto_pagado,
    r.comision,
    r.estado,
    a.codigo            AS codigo_afiliado,
    ua.email            AS afiliado_email,
    a.saldo             AS saldo_afiliado_actual
FROM   referidos  r
JOIN   afiliados  a  ON a.id  = r.afiliado_id
JOIN   auth.users ua ON ua.id = a.user_id
ORDER  BY r.creado_en DESC;

-- Bloquear acceso via API (DEFAULT PRIVILEGES de Supabase auto-concede SELECT a nuevas vistas)
REVOKE SELECT ON v_referidos_admin FROM PUBLIC, anon, authenticated;


-- ────────────────────────────────────────────────────────────────
-- 9. QUERIES DE TRABAJO DIARIO
--    Copiar y pegar en el SQL Editor segun necesidad
-- ────────────────────────────────────────────────────────────────

/*

-- Ver pendientes para aprobar (cruzar con Meru por comprador_email):
SELECT referido_id, fecha, comprador_email, plan, monto_pagado, comision,
       codigo_afiliado, afiliado_email
FROM   v_referidos_admin
WHERE  estado = 'pendiente'
ORDER  BY fecha ASC;

-- Aprobar una comision (pegar el referido_id de la query anterior):
SELECT aprobar_comision('xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx');

-- Ver retiros pendientes de pagar:
SELECT r.id        AS retiro_id,
       r.creado_en,
       r.monto,
       r.metodo,
       r.datos_pago,
       ua.email    AS afiliado_email
FROM   retiros r
JOIN   afiliados  a  ON a.id  = r.afiliado_id
JOIN   auth.users ua ON ua.id = a.user_id
WHERE  r.estado = 'pendiente'
ORDER  BY r.creado_en ASC;

-- Pagar un retiro:
SELECT procesar_retiro('xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx', true);

-- Rechazar un retiro (restaura saldo automaticamente):
SELECT procesar_retiro('xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx', false);

-- Modificar precio o comision de un pack (solo desde SQL Editor):
UPDATE packs SET precio_usd = 14.99, actualizado_en = now() WHERE id = 'pro';

*/


-- ────────────────────────────────────────────────────────────────
-- 10. VERIFICACION — ejecutar despues para confirmar
-- ────────────────────────────────────────────────────────────────

-- Tablas creadas
SELECT table_name
FROM   information_schema.tables
WHERE  table_schema = 'public'
  AND  table_name IN ('packs','afiliados','referidos','retiros')
ORDER  BY table_name;

-- Precios cargados correctamente
SELECT * FROM packs ORDER BY precio_usd;

-- Firmas de las 5 funciones
SELECT proname, pg_get_function_arguments(oid) AS firma
FROM   pg_proc
WHERE  proname IN (
    'registrar_afiliado',
    'registrar_referido',
    'solicitar_retiro',
    'aprobar_comision',
    'procesar_retiro'
)
ORDER  BY proname;

-- Grants de seguridad
-- aprobar_comision y procesar_retiro deben ser false para authenticated
SELECT
    p.proname                                           AS funcion,
    r.rolname                                           AS rol,
    has_function_privilege(r.rolname, p.oid, 'EXECUTE') AS puede_ejecutar
FROM   pg_proc  p
CROSS  JOIN pg_roles r
WHERE  p.proname IN (
    'registrar_afiliado',
    'registrar_referido',
    'solicitar_retiro',
    'aprobar_comision',
    'procesar_retiro'
)
  AND  r.rolname IN ('anon','authenticated','service_role')
ORDER  BY p.proname, r.rolname;

-- Politicas RLS activas
SELECT tablename, policyname, cmd AS operacion, roles
FROM   pg_policies
WHERE  tablename IN ('packs','afiliados','referidos','retiros')
ORDER  BY tablename, policyname;
