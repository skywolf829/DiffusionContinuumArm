import torch
from typing import Tuple

def kinematics(phi: torch.Tensor, theta: torch.Tensor, xi: torch.Tensor, L: float = 0.15) -> torch.Tensor:
    """Compute the section kinematics as a homogeneous transform.

    This function is written to be fully tensorized / batched in PyTorch.

    Args:
        phi:   (...,) bending plane parameter
        theta: (...,) bending magnitude parameter
        xi:    (...,) normalized arc-length parameter in [0, 1] (or broadcastable)
        L: section length

    Returns:
        T: (..., 4, 4) homogeneous transform
    """
    # Ensure tensors are on the same device/dtype and broadcastable
    phi, theta, xi = torch.broadcast_tensors(phi, theta, xi)
    dtype = phi.dtype
    device = phi.device

    # Use tensor scalar for L so gradients / devices behave nicely
    L_t = torch.as_tensor(L, dtype=dtype, device=device)

    # Precompute powers that are reused a lot
    xi2 = xi * xi
    xi4 = xi2 * xi2
    xi6 = xi4 * xi2
    xi8 = xi4 * xi4
    xi10 = xi8 * xi2
    xi12 = xi6 * xi6
    xi14 = xi12 * xi2
    xi16 = xi8 * xi8
    xi18 = xi16 * xi2

    phi2 = phi * phi
    phi4 = phi2 * phi2
    phi6 = phi4 * phi2
    phi8 = phi4 * phi4
    phi10 = phi8 * phi2
    phi12 = phi6 * phi6
    phi14 = phi12 * phi2
    phi16 = phi8 * phi8
    phi18 = phi16 * phi2

    th2 = theta * theta
    th4 = th2 * th2
    th6 = th4 * th2
    th8 = th4 * th4
    th10 = th8 * th2
    th12 = th6 * th6
    th14 = th12 * th2
    th16 = th8 * th8
    th18 = th16 * th2

    # Allocate output with batch dims
    out_shape = phi.shape + (4, 4)
    res = torch.zeros(out_shape, device=device, dtype=dtype)

    # --- Matrix entries (ported from your symbolic expansion) ---
    # NOTE: these expressions are intentionally kept algebraically identical,
    # but computed with tensor ops and batched shapes.

    res[..., 0, 0] = (
        1
        + xi12 * phi12 / 479001600
        + (th2 - 1) * xi10 * phi10 / 3628800
        + xi8 * (th4 - 3 * th2 + 3) * phi8 / 120960
        + xi6 * (th6 - 7.5 * th4 + 22.5 * th2 - 22.5) * phi6 / 16200
        + xi4 * (th8 - 14 * th6 + 105 * th4 - 315 * th2 + 315) * phi4 / 7560
        + (th10 - 22.5 * th8 + 315 * th6 - 2362.5 * th4 + 7087.5 * th2 - 7087.5) * xi2 * phi2 / 14175
    )

    res[..., 0, 1] = (
        -xi2
        * theta
        * phi2
        * (
            xi8 * phi8
            + 60 * phi6 * th2 * xi6
            - 90 * phi6 * xi6
            + 672 * phi4 * th4 * xi4
            - 3360 * phi4 * th2 * xi4
            + 1920 * phi2 * th6 * xi2
            + 5040 * phi4 * xi4
            - 20160 * phi2 * th4 * xi2
            + 1280 * th8
            + 100800 * phi2 * th2 * xi2
            - 23040 * th6
            - 151200 * phi2 * xi2
            + 241920 * th4
            - 1209600 * th2
            + 1814400
        )
        / 3628800
    )

    res[..., 0, 2] = (
        -xi
        * phi
        * (
            -39916800
            + xi10 * phi10
            + 55 * phi8 * th2 * xi8
            + 330 * phi6 * th4 * xi6
            + 462 * phi4 * th6 * xi4
            + 165 * phi2 * th8 * xi2
            + 11 * th10
            - 110
            * (phi2 * xi2 + 3 * th2)
            * (phi6 * xi6 + 33 * phi4 * th2 * xi4 + 27 * phi2 * th4 * xi2 + 3 * th6)
            + 7920 * phi6 * xi6
            + 166320 * phi4 * th2 * xi4
            + 277200 * phi2 * th4 * xi2
            + 55440 * th6
            - 332640 * phi4 * xi4
            - 3326400 * phi2 * th2 * xi2
            - 1663200 * th4
            + 6652800 * phi2 * xi2
            + 19958400 * th2
        )
        / 39916800
    )

    res[..., 0, 3] = (
        -phi
        * xi2
        * (
            xi10 * phi10
            + 66 * phi8 * th2 * xi8
            - 132 * xi8 * phi8
            + 495 * phi6 * th4 * xi6
            - 5940 * phi6 * th2 * xi6
            + 924 * phi4 * th6 * xi4
            + 11880 * phi6 * xi6
            - 27720 * phi4 * th4 * xi4
            + 495 * phi2 * th8 * xi2
            + 332640 * phi4 * th2 * xi4
            - 27720 * phi2 * th6 * xi2
            + 66 * th10
            - 665280 * phi4 * xi4
            + 831600 * phi2 * th4 * xi2
            - 5940 * th8
            - 9979200 * phi2 * th2 * xi2
            + 332640 * th6
            + 19958400 * phi2 * xi2
            - 9979200 * th4
            + 119750400 * th2
            - 239500800
        )
        * L_t
        / 479001600
    )

    res[..., 1, 0] = (
        -xi2
        * theta
        * (
            xi16 * phi16
            + 204 * phi14 * th2 * xi14
            - 306 * phi14 * xi14
            + 9792 * phi12 * th4 * xi12
            - 48960 * phi12 * th2 * xi12
            + 169728 * phi10 * th6 * xi10
            + 73440 * xi12 * phi12
            - 1782144 * phi10 * th4 * xi10
            + 1244672 * phi8 * th8 * xi8
            + 8910720 * phi10 * th2 * xi10
            - 22404096 * phi8 * th6 * xi8
            + 4073472 * phi6 * th10 * xi6
            - 13366080 * xi10 * phi10
            + 235243008 * phi8 * th4 * xi8
            - 112020480 * phi6 * th8 * xi6
            + 5849088 * phi4 * th12 * xi4
            - 1176215040 * phi8 * th2 * xi8
            + 2016368640 * phi6 * th6 * xi6
            - 228114432 * phi4 * th10 * xi4
            + 3342336 * phi2 * th14 * xi2
            + 1764322560 * xi8 * phi8
            - 21171870720 * phi6 * th4 * xi6
            + 6273146880 * phi4 * th8 * xi4
            - 175472640 * phi2 * th12 * xi2
            + 589824 * th16
            + 105859353600 * phi6 * th2 * xi6
            - 112916643840 * phi4 * th6 * xi4
            + 6843432960 * th10 * phi2 * xi2
            - 40108032 * th14
            - 158789030400 * phi6 * xi6
            + 1185624760320 * phi4 * th4 * xi4
            - 188194406400 * phi2 * th8 * xi2
            + 2105671680 * th12
            - 5928123801600 * phi4 * th2 * xi4
            + 3387499315200 * phi2 * th6 * xi2
            - 82121195520 * th10
            + 8892185702400 * phi4 * xi4
            - 35568742809600 * phi2 * th4 * xi2
            + 2258332876800 * th8
            + 177843714048000 * phi2 * th2 * xi2
            - 40649991782400 * th6
            - 266765571072000 * phi2 * xi2
            + 426824913715200 * th4
            - 2134124568576000 * th2
            + 3201186852864000
        )
        * phi2
        / 6402373705728000
    )

    res[..., 1, 1] = (
        1
        - th10 * phi2 * xi2 / 14175
        - xi2 * phi2 * (phi2 * xi2 - 12) * th8 / 7560
        - xi2 * phi2 * (phi4 * xi4 - 30 * phi2 * xi2 + 360) * th6 / 16200
        - xi2 * phi2 * (phi6 * xi6 - 56 * phi4 * xi4 + 1680 * phi2 * xi2 - 20160) * th4 / 120960
        - xi2 * phi2 * (xi8 * phi8 - 90 * phi6 * xi6 + 5040 * phi4 * xi4 - 151200 * phi2 * xi2 + 1814400) * th2 / 3628800
    )

    res[..., 1, 2] = (
        -xi
        * theta
        * phi
        * (
            -39916800
            + xi10 * phi10
            + (5.5 / 3.0) * phi8 * th2 * xi8
            + 66 * phi6 * th4 * xi6
            + 66 * phi4 * th6 * xi4
            + (5.5 / 3.0) * phi2 * th8 * xi2
            + th10
            - 110
            * (phi4 * xi4 + 2 * phi2 * th2 * xi2 + th4 / 5)
            * (phi4 * xi4 + 10 * phi2 * th2 * xi2 + 5 * th4)
            + 7920
            * (phi2 * xi2 + th2)
            * (phi4 * xi4 + 6 * phi2 * th2 * xi2 + th4)
            - 332640 * phi4 * xi4
            - 1108800 * phi2 * th2 * xi2
            - 332640 * th4
            + 6652800 * phi2 * xi2
            + 6652800 * th2
        )
        / 39916800
    )

    res[..., 1, 3] = (
        phi
        * xi2
        * (
            xi8 * phi8
            + 15 * phi6 * th2 * xi6
            - 90 * phi6 * xi6
            + 42 * phi4 * th4 * xi4
            - 840 * phi4 * th2 * xi4
            + 30 * phi2 * th6 * xi2
            + 5040 * phi4 * xi4
            - 1260 * phi2 * th4 * xi2
            + 5 * th8
            + 25200 * phi2 * th2 * xi2
            - 360 * th6
            - 151200 * phi2 * xi2
            + 15120 * th4
            - 302400 * th2
            + 1814400
        )
        * theta
        * L_t
        / 3628800
    )

    res[..., 2, 0] = (
        xi
        * (
            -121645100408832000
            - 1013709170073600 * phi4 * xi4
            + 20274183401472000 * phi2 * xi2
            + 24135932620800 * phi6 * xi6
            + 506854585036800 * phi4 * th2 * xi4
            + 844757641728000 * phi2 * th4 * xi2
            - 10137091700736000 * phi2 * th2 * xi2
            - 253955520 * th12
            + 93024
            * (phi2 * xi2 + 3 * th2)
            * (phi4 * xi4 + 10 * phi2 * th2 * xi2 + 5 * th4)
            * (xi8 * phi8 + 92 * phi6 * th2 * xi6 + 134 * phi4 * th4 * xi4 + 28 * phi2 * th6 * xi2 + th8)
            + 3047466240 * xi10 * phi10
            - 5814 * th16
            + 19 * th18
            - 335221286400
            * (phi2 * xi2 + 3 * th2)
            * (phi6 * xi6 + 33 * phi4 * th2 * xi4 + 27 * phi2 * th4 * xi2 + 3 * th6)
            + 167610643200 * phi8 * th2 * xi8
            + 1005663859200 * phi6 * th4 * xi6
            + 1407929402880 * phi4 * th6 * xi4
            + 502831929600 * phi2 * th8 * xi2
            + 171 * phi16 * th2 * xi16
            + 3876 * phi14 * th4 * xi14
            + 27132 * phi12 * th6 * xi12
            + 75582 * phi10 * th8 * xi10
            + 92378 * phi8 * th10 * xi8
            + 50388 * phi6 * th12 * xi6
            + 11628 * phi4 * th14 * xi4
            + 969 * phi2 * th16 * xi2
            - 46512 * phi14 * th2 * xi14
            - 813960 * phi12 * th4 * xi12
            - 4232592 * phi10 * th6 * xi10
            - 8314020 * phi8 * th8 * xi8
            - 6651216 * phi6 * th10 * xi6
            - 2116296 * phi4 * th12 * xi4
            - 232560 * phi2 * th14 * xi2
            - 1523733120 * phi10 * th2 * xi10
            - 13967553600 * phi8 * th4 * xi8
            - 33522128640 * phi6 * th6 * xi6
            - 25141596480 * phi4 * th8 * xi4
            - 5587021440 * th10 * phi2 * xi2
            - 342 * xi16 * phi16
            + xi18 * phi18
            + 60822550204416000 * th2
            - 5068545850368000 * th4
            + 168951528345600 * th6
            + 33522128640 * th10
            - 19535040 * xi12 * phi12
        )
        * phi
        / 121645100408832000
    )

    res[..., 2, 1] = (
        -xi
        * theta
        * phi
        * (
            355687428096000
            + (phi6 * xi6 + 9 * phi4 * th2 * xi4 + 11 * phi2 * th4 * xi2 + th6 / 3)
            * (phi6 * xi6 + 33 * phi4 * th2 * xi4 + 27 * phi2 * th4 * xi2 + 3 * th6)
            * (phi2 * xi2 + th2 / 3)
            * (phi2 * xi2 + 3 * th2)
            - 272
            * (phi2 * xi2 + th2)
            * (phi4 * xi4 + 6 * phi2 * th2 * xi2 + th4)
            * (xi8 * phi8 + 28 * phi6 * th2 * xi6 + 70 * phi4 * th4 * xi4 + 28 * phi2 * th6 * xi2 + th8)
            + 57120
            * (phi6 * xi6 + 21 * phi4 * th2 * xi4 + 35 * phi2 * th4 * xi2 + 7 * th6)
            * (phi6 * xi6 + 5 * phi4 * th2 * xi4 + 3 * phi2 * th4 * xi2 + th6 / 7)
            - 8910720
            * (phi4 * xi4 + 14 * phi2 * th2 * xi2 + th4)
            * (phi2 * xi2 + th2)
            * (phi2 * xi2 + th2 / 3)
            * (phi2 * xi2 + 3 * th2)
            + 980179200
            * (phi4 * xi4 + 2 * phi2 * th2 * xi2 + th4 / 5)
            * (phi4 * xi4 + 10 * phi2 * th2 * xi2 + 5 * th4)
            - 70572902400 * (phi2 * xi2 + th2) * (phi4 * xi4 + 6 * phi2 * th2 * xi2 + th4)
            + 2964061900800 * (phi2 * xi2 + th2 / 3) * (phi2 * xi2 + 3 * th2)
            - 59281238016000 * phi2 * xi2
            - 59281238016000 * th2
        )
        / 355687428096000
    )

    res[..., 2, 2] = (
        1
        - phi2 * xi2 / 2
        + phi4 * xi4 / 24
        - phi6 * xi6 / 720
        + xi8 * phi8 / 40320
        - xi10 * phi10 / 3628800
        + xi12 * phi12 / 479001600
    )

    res[..., 2, 3] = (
        -xi
        * L_t
        * (xi10 * phi10 - 110 * xi8 * phi8 + 7920 * phi6 * xi6 - 332640 * phi4 * xi4 + 6652800 * phi2 * xi2 - 39916800)
        / 39916800
    )

    res[..., 3, 3] = 1

    return res


def section_kinematics(config: torch.Tensor, section: int, xi: torch.Tensor, L: float = 0.15) -> torch.Tensor:
    """Compute the Cartesian position for a given section/xi.

    Args:
        config: (..., 6) packed as [phi0, theta0, phi1, theta1, phi2, theta2]
        section: 0, 1, or 2 (which segment end is being queried)
        xi: (...,) normalized arc-length in [0,1] for the queried section
        L: section length

    Returns:
        pos: (..., 3) translation component of the composed transform
    """
    if config.shape[-1] != 6:
        raise ValueError(f"config must have last dim 6, got {config.shape}")

    # Unpack as phi/theta pairs per segment (matching your original code usage)
    phi0, th0, phi1, th1, phi2, th2 = [config[..., i] for i in range(6)]

    if section == 0:
        m = kinematics(phi0, th0, xi, L=L)
    elif section == 1:
        m1 = kinematics(phi0, th0, torch.ones_like(xi), L=L)
        m2 = kinematics(phi1, th1, xi, L=L)
        m = torch.matmul(m1, m2)
    elif section == 2:
        m1 = kinematics(phi0, th0, torch.ones_like(xi), L=L)
        m2 = kinematics(phi1, th1, torch.ones_like(xi), L=L)
        m3 = kinematics(phi2, th2, xi, L=L)
        m = torch.matmul(torch.matmul(m1, m2), m3)
    else:
        raise ValueError(f"section must be 0, 1, or 2; got {section}")

    return m[..., 0:3, 3]

def continuum_fk_points(
    x_phys: torch.Tensor,
    points_per_segment: int = 10,
    L: float = 0.15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched FK wrapper compatible with diffusion-MPC pipeline.

    Args:
        x_phys: (B, T, 6) [phi0, theta0, phi1, theta1, phi2, theta2]
        points_per_segment: number of xi samples per section
        L: section length

    Returns:
        points: (B, T, P, 3)
        tip:    (B, T, 3)
    """
    B, T, _ = x_phys.shape
    device = x_phys.device
    dtype = x_phys.dtype

    # xi samples in [0,1]
    xi = torch.linspace(0.0, 1.0, points_per_segment, device=device, dtype=dtype)

    all_points = []

    for section in (0, 1, 2):
        # Broadcast xi to (B,T,P)
        xi_bt = xi.view(1, 1, -1).expand(B, T, -1)

        # section_kinematics returns (..., 3)
        pts = section_kinematics(
            config=x_phys.unsqueeze(2).expand(B, T, points_per_segment, 6),
            section=section,
            xi=xi_bt,
            L=L,
        )  # (B,T,P,3)

        all_points.append(pts)

    points = torch.cat(all_points, dim=2)  # (B,T,3*P,3)

    # Tip = end of section 2 at xi=1
    tip = section_kinematics(
        config=x_phys,
        section=2,
        xi=torch.ones((B, T), device=device, dtype=dtype),
        L=L,
    )  # (B,T,3)

    return points, tip
