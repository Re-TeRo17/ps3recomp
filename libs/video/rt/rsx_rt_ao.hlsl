// RTAO compute shaders (SM 6.5, inline ray queries). Compiled OFFLINE with dxc
// (rt/compile_rt_shaders.ps1) into rsx_rt_ao_dxil.h so that no DXC runtime is shipped.
//
// Root layout (see rsx_rt.c): b0 = 60 root constants, one descriptor table
// { t0..t4 SRV, u0..u1 UAV }.
//   trace   : t0 depth, t1 static TLAS, t3 dynamic TLAS   -> u0 raw AO (low res)
//   temporal: t0 depth, t2 raw AO, t4 history (previous)  -> u0 accumulated AO, u1 history (current: ao, dist, count)
//   blur_h  : t0 depth, t2 accumulated AO                 -> u0 tmp (low res)
//   up      : t0 depth, t2 tmp                            -> u0 final AO (full res)

cbuffer P : register(b0)
{
    float4 minv0;       // inverse of the camera-relative world-to-clip matrix, one row per float4
    float4 minv1;
    float4 minv2;
    float4 minv3;
    float4 eye_radius;  // xyz: the reference point E (world), w: ray length
    float4 vpx;         // xf0, xf1, xf4, xf5: clip/w -> ndc.xy
    float4 vpz;         // xf2, xf6: ndc depth mapping; z: strength; w: bias
    uint4  dims;        // low-res w,h ; full-res w,h
    uint4  cfg;         // frame, rays, scale (1|2), debug
    float4 pm0;         // previous frame's camera-relative world-to-clip matrix (relative to ITS reference point)
    float4 pm1;
    float4 pm2;
    float4 pm3;
    float4 pdelta;      // xyz: E_now - E_prev, w: 1 when the history is usable
    float4 ext;         // x: power, y: minimum history weight, z: 1 when the dynamic TLAS has instances, w: AO weight on pixels the static TLAS does not explain (characters)
};

Texture2D<float>               gdepth : register(t0);
RaytracingAccelerationStructure gtlas : register(t1);
Texture2D<float>               gsrc   : register(t2);
RaytracingAccelerationStructure gdtlas : register(t3);
Texture2D<float4>              ghist  : register(t4);
RWTexture2D<float>             gout   : register(u0);
RWTexture2D<float4>            ghout  : register(u1);

// World position relative to E of the surface seen at full-res pixel p with depth d.
float3 Recon(int2 p, float d)
{
    float2 uv = (float2(p) + 0.5) / float2(dims.zw);
    float cx = ((uv.x * 2.0 - 1.0) - vpx.z) / vpx.x;
    float cy = ((1.0 - uv.y * 2.0) - vpx.w) / vpx.y;
    float cz = (d - vpz.y) / vpz.x;
    float4 c = float4(cx, cy, cz, 1.0);
    float4 h = float4(dot(minv0, c), dot(minv1, c), dot(minv2, c), dot(minv3, c));
    return h.xyz / h.w;
}

uint Pcg(uint v) { uint st = v * 747796405u + 2891336453u; uint w = ((st >> ((st >> 28u) + 4u)) ^ st) * 277803737u; return (w >> 22u) ^ w; }
float IGN(float2 p) { return frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715)))); }

bool IsSky(float d) { return d >= 0.999999 || d <= 0.0; }

int2 ClampP(int2 p) { return clamp(p, int2(0, 0), int2(dims.zw) - 1); }

[numthreads(8, 8, 1)]
void cs_trace(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= dims.x || id.y >= dims.y) return;
    uint scale = cfg.z;
    int2 fp = ClampP(int2(id.xy) * int(scale) + int(scale >> 1));
    float d = gdepth.Load(int3(fp, 0));
    float vis = 1.0;
    if (!IsSky(d))
    {
        float3 P = Recon(fp, d);
        float lenP = length(P);
        if (cfg.w <= 1u && lenP > 120.0) { gout[id.xy] = 1.0; return; }     // beyond the AO fade: nothing to trace
        if (cfg.w == 2u)
        {
            // Debug: a ray from the reference point to the pixel's surface must hit the
            // static scene at about the same distance (1 match, 0.5 other hit, 0 miss).
            RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
            RayDesc r; r.Origin = eye_radius.xyz; r.Direction = P / lenP; r.TMin = 0.0; r.TMax = lenP * 1.2 + 1.0;
            q.TraceRayInline(gtlas, 0, 1, r);
            while (q.Proceed()) {}
            vis = 0.0;
            if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
                vis = (abs(q.CommittedRayT() - lenP) < 0.03 * lenP + 0.05) ? 1.0 : 0.5;
            if (vis < 1.0 && ext.z > 0.5)
            {
                RayQuery<RAY_FLAG_FORCE_OPAQUE> q2;
                q2.TraceRayInline(gdtlas, 0, 0xFF, r);
                while (q2.Proceed()) {}
                if (q2.CommittedStatus() == COMMITTED_TRIANGLE_HIT && abs(q2.CommittedRayT() - lenP) < 0.03 * lenP + 0.05) vis = 1.0;
            }
            gout[id.xy] = vis;
            return;
        }
        if (cfg.w == 3u)
        {
            // Debug: the scene's world position, drawn as a checker of 1 x 1 x 1 world-unit cubes (scale check).
            float3 Wp = eye_radius.xyz + P;
            int3 cell = int3(floor(Wp));
            gout[id.xy] = (((cell.x + cell.y + cell.z) & 1) != 0) ? 0.9 : 0.3;
            return;
        }
        float dl = gdepth.Load(int3(ClampP(fp + int2(-1, 0)), 0));
        float dr = gdepth.Load(int3(ClampP(fp + int2( 1, 0)), 0));
        float du = gdepth.Load(int3(ClampP(fp + int2( 0,-1)), 0));
        float dd = gdepth.Load(int3(ClampP(fp + int2( 0, 1)), 0));
        float3 Pl = Recon(ClampP(fp + int2(-1, 0)), dl);
        float3 Pr = Recon(ClampP(fp + int2( 1, 0)), dr);
        float3 Pu = Recon(ClampP(fp + int2( 0,-1)), du);
        float3 Pd = Recon(ClampP(fp + int2( 0, 1)), dd);
        float3 dx = (abs(dr - d) < abs(dl - d)) ? (Pr - P) : (P - Pl);
        float3 dy = (abs(dd - d) < abs(du - d)) ? (Pd - P) : (P - Pu);
        {
            // On a smooth surface take a 4-pixel central difference: the depth steps of a wall seen at a grazing angle
            // do not turn into bands of normals.
            float tolN = 0.015 * lenP + 0.02;
            float3 Pl2 = Recon(ClampP(fp + int2(-2, 0)), gdepth.Load(int3(ClampP(fp + int2(-2, 0)), 0)));
            float3 Pr2 = Recon(ClampP(fp + int2( 2, 0)), gdepth.Load(int3(ClampP(fp + int2( 2, 0)), 0)));
            float3 Pu2 = Recon(ClampP(fp + int2(0, -2)), gdepth.Load(int3(ClampP(fp + int2(0, -2)), 0)));
            float3 Pd2 = Recon(ClampP(fp + int2(0,  2)), gdepth.Load(int3(ClampP(fp + int2(0,  2)), 0)));
            if (abs(length(Pr) - lenP) < tolN && abs(length(Pl) - lenP) < tolN && abs(length(Pr2) - lenP) < 2.0 * tolN && abs(length(Pl2) - lenP) < 2.0 * tolN)
                dx = Pr2 - Pl2;
            if (abs(length(Pd) - lenP) < tolN && abs(length(Pu) - lenP) < tolN && abs(length(Pd2) - lenP) < 2.0 * tolN && abs(length(Pu2) - lenP) < 2.0 * tolN)
                dy = Pd2 - Pu2;
        }
        float3 N = cross(dy, dx);
        float nl = length(N);
        if (nl > 1e-12)
        {
            N /= nl;
            if (dot(N, P) > 0.0) N = -N;
            float radius = eye_radius.w;
            // Does the static scene (instances of still geometry, mask 1) contain this pixel's surface? Characters and
            // animated props (mask 2 / the per-frame TLAS) and geometry the TLAS lacks do not. Those pixels get a weaker
            // AO, from a smoothed normal (the depth-derived one is noisy on fur) and from the static scene only, so that
            // nothing can occlude itself with a stale pose.
            float att = 1.0;
            bool still = true;
            if (ext.w < 0.999)
            {
                // a short ray across the surface along the view direction (cheap: the long eye ray cost 3 ms at 4K)
                float dlt = 0.03 * lenP + 0.05;
                float3 vd = P / lenP;
                RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> pq;
                RayDesc pr; pr.Origin = eye_radius.xyz + P - vd * dlt; pr.Direction = vd; pr.TMin = 0.0; pr.TMax = 2.0 * dlt;
                pq.TraceRayInline(gtlas, 0, 1, pr);
                while (pq.Proceed()) {}
                still = pq.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
                att = still ? 1.0 : ext.w;
                if (!still) N = normalize(lerp(N, float3(0, 0, 1), 0.75));
            }
            float3 up = abs(N.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
            float3 T = normalize(cross(up, N));
            float3 B = cross(N, T);
            float3 O = eye_radius.xyz + P + N * ((0.03 + 0.006 * lenP) / max(0.3, abs(dot(N, P / lenP))));
            uint rays = lenP > 40.0 ? max(1u, cfg.y / 2u) : cfg.y;                // fewer rays far away
            uint hsh = Pcg(id.x + Pcg(id.y + Pcg(cfg.x)));            // cfg.x = 0 unless RT_AO_ANIM (the history needs a new pattern every frame)
            float s1 = float(hsh & 0xFFFFu) / 65536.0, s2 = float(hsh >> 16) / 65536.0;
            float occ = 0.0;
            for (uint k = 0; k < rays; k++)
            {
                float idx = float(k);
                float u1 = frac(0.5 + idx * 0.7548776662 + s1);
                float u2 = frac(0.5 + idx * 0.5698402910 + s2);
                float rr = sqrt(u1), ph = 6.2831853 * u2;
                float3 L = T * (rr * cos(ph)) + B * (rr * sin(ph)) + N * sqrt(1.0 - u1);
                RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
                RayDesc r; r.Origin = O; r.Direction = L; r.TMin = 0.0; r.TMax = radius;
                q.TraceRayInline(gtlas, 0, still ? 0xFF : 1, r);
                while (q.Proceed()) {}
                if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
                {
                    float t = saturate(q.CommittedRayT() / radius);
                    occ += 1.0 - t * t;
                }
                else if (still && ext.z > 0.5)
                {
                    // animated geometry (characters, skinned props), rebuilt every frame
                    RayQuery<RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q2;
                    q2.TraceRayInline(gdtlas, 0, 0xFF, r);
                    while (q2.Proceed()) {}
                    if (q2.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
                    {
                        float t = saturate(q2.CommittedRayT() / radius);
                        occ += 1.0 - t * t;
                    }
                }
            }
            vis = lerp(1.0, 1.0 - occ / float(rays), att);
        }
    }
    gout[id.xy] = vis;
}

float LinDepth(int2 p, float d) { return length(Recon(p, d)); }

// Temporal accumulation (low res): reproject the pixel into the previous frame, take the history
// where the stored distance from the eye agrees (depth rejection), blend with the new rays.
[numthreads(8, 8, 1)]
void cs_temporal(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= dims.x || id.y >= dims.y) return;
    int scale = int(cfg.z);
    int2 fp = ClampP(int2(id.xy) * scale + (scale >> 1));
    float d = gdepth.Load(int3(fp, 0));
    float cur = gsrc.Load(int3(id.xy, 0));
    if (IsSky(d)) { gout[id.xy] = 1.0; ghout[id.xy] = float4(1.0, 0.0, 0.0, 0.0); return; }
    float3 P = Recon(fp, d);
    float dist = length(P);
    float hist = cur, cnt = 0.0;
    if (pdelta.w > 0.5)
    {
        float3 Pp = P + pdelta.xyz;
        float expd = length(Pp);
        float4 c = float4(dot(pm0, float4(Pp, 1.0)), dot(pm1, float4(Pp, 1.0)), dot(pm2, float4(Pp, 1.0)), dot(pm3, float4(Pp, 1.0)));
        if (c.w > 1e-4)
        {
            float2 ndc = c.xy / c.w * vpx.xy + vpx.zw;
            float2 uv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
            if (all(uv > 0.0) && all(uv < 1.0))
            {
                float2 lp = uv * float2(dims.xy) - 0.5;
                int2 b = int2(floor(lp));
                float2 f = lp - floor(lp);
                float sum = 0.0, wsum = 0.0, csum = 0.0;
                [unroll] for (int j = 0; j < 2; j++)
                [unroll] for (int i = 0; i < 2; i++)
                {
                    int2 t = clamp(b + int2(i, j), int2(0, 0), int2(dims.xy) - 1);
                    float4 h = ghist.Load(int3(t, 0));
                    float wb = (i == 0 ? 1.0 - f.x : f.x) * (j == 0 ? 1.0 - f.y : f.y);
                    float ok = (h.z > 0.5 && abs(h.y - expd) < 0.05 * expd + 0.04) ? 1.0 : 0.0;
                    float w = wb * ok;
                    sum += w * h.x; csum += w * h.z; wsum += w;
                }
                if (wsum > 0.25) { hist = sum / wsum; cnt = csum / wsum; }
            }
        }
    }
    float alpha = (cnt < 0.5) ? 1.0 : max(ext.y, 1.0 / (cnt + 1.0));
    float res = lerp(hist, cur, alpha);
    gout[id.xy] = res;
    ghout[id.xy] = float4(res, dist, min(cnt + 1.0, 64.0), 0.0);
}

[numthreads(8, 8, 1)]
void cs_blur_h(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= dims.x || id.y >= dims.y) return;
    int scale = int(cfg.z);
    int2 fpc = ClampP(int2(id.xy) * scale + (scale >> 1));
    float dc = gdepth.Load(int3(fpc, 0));
    if (IsSky(dc)) { gout[id.xy] = 1.0; return; }
    float lc = LinDepth(fpc, dc);
    // a cross of nine taps, both directions (a horizontal-only filter leaves horizontal streaks)
    static const int2 offs[9] = { int2(0, 0), int2(-1, 0), int2(1, 0), int2(-2, 0), int2(2, 0), int2(0, -1), int2(0, 1), int2(0, -2), int2(0, 2) };
    float sum = 0.0, wsum = 0.0;
    [unroll] for (int k = 0; k < 9; k++)
    {
        int2 lp = clamp(int2(id.xy) + offs[k], int2(0, 0), int2(dims.xy) - 1);
        int2 fp = ClampP(lp * scale + (scale >> 1));
        float d = gdepth.Load(int3(fp, 0));
        float v = gsrc.Load(int3(lp, 0));
        float l = IsSky(d) ? 1e9 : LinDepth(fp, d);
        float w = exp(-0.25 * float(dot(offs[k], offs[k]))) * exp(-abs(l - lc) / (0.04 * lc + 0.03));
        sum += w * v; wsum += w;
    }
    gout[id.xy] = sum / max(wsum, 1e-5);
}

[numthreads(8, 8, 1)]
void cs_up(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= dims.z || id.y >= dims.w) return;
    int scale = int(cfg.z);
    int2 fp = int2(id.xy);
    float d = gdepth.Load(int3(fp, 0));
    if (IsSky(d)) { gout[id.xy] = 1.0; return; }
    float3 P = Recon(fp, d);
    float lc = length(P);
    // normal of this pixel (the smaller depth step on each axis)
    float dl = gdepth.Load(int3(ClampP(fp + int2(-1, 0)), 0));
    float dr = gdepth.Load(int3(ClampP(fp + int2( 1, 0)), 0));
    float du = gdepth.Load(int3(ClampP(fp + int2( 0,-1)), 0));
    float dd = gdepth.Load(int3(ClampP(fp + int2( 0, 1)), 0));
    float3 Pl = Recon(ClampP(fp + int2(-1, 0)), dl), Pr = Recon(ClampP(fp + int2(1, 0)), dr);
    float3 Pu = Recon(ClampP(fp + int2(0, -1)), du), Pd = Recon(ClampP(fp + int2(0, 1)), dd);
    float3 gx = (abs(dr - d) < abs(dl - d)) ? (Pr - P) : (P - Pl);
    float3 gy = (abs(dd - d) < abs(du - d)) ? (Pd - P) : (P - Pu);
    float3 Nf = cross(gy, gx);
    float nlen = length(Nf);
    Nf = nlen > 1e-12 ? Nf / nlen : float3(0, 0, 1);
    float2 pos = (float2(fp) + 0.5) / float(scale) - 0.5;   // in low-res pixel units
    int2 b0 = int2(floor(pos));
    float2 fr = pos - floor(pos);
    float sum = 0.0, wsum = 0.0;
    float tol = 0.015 * lc + 0.02;
    [unroll] for (int j = 0; j <= 1; j++)
    [unroll] for (int i = 0; i <= 1; i++)
    {
        int2 lp = clamp(b0 + int2(i, j), int2(0, 0), int2(dims.xy) - 1);
        float wt = (i == 0 ? 1.0 - fr.x : fr.x) * (j == 0 ? 1.0 - fr.y : fr.y);
        int2 sp = ClampP(lp * scale + (scale >> 1));
        float ds = gdepth.Load(int3(sp, 0));
        // distance of the sample's surface point from the tangent plane of this pixel: depth AND orientation
        float pd = IsSky(ds) ? 1e9 : abs(dot(Recon(sp, ds) - P, Nf));
        float w = wt * exp(-pd / tol) + 1e-4 * wt;
        sum += w * gsrc.Load(int3(lp, 0)); wsum += w;
    }
    float v = sum / max(wsum, 1e-6);
    if (cfg.w <= 1u)                                        // 1 = debug view of the term that is multiplied
    {
        float fade = saturate((lc - 60.0) / 60.0);          // far geometry: no AO
        v = pow(saturate(v), ext.x);                        // contrast
        v = saturate(lerp(lerp(1.0, v, vpz.z), 1.0, fade));
        // silhouettes: the colour of an edge pixel mixes surfaces (MSAA, thin fur), the AO is one number per pixel
        float e = 0.0;
        e = max(e, IsSky(dl) ? 9.0 : abs(length(Pl) - lc));
        e = max(e, IsSky(dr) ? 9.0 : abs(length(Pr) - lc));
        e = max(e, IsSky(du) ? 9.0 : abs(length(Pu) - lc));
        e = max(e, IsSky(dd) ? 9.0 : abs(length(Pd) - lc));
        float edge = saturate((e / (0.03 * lc + 0.05) - 1.0) / 2.0);
        v = lerp(v, 1.0, edge * 0.85);
    }
    gout[id.xy] = v;
}
