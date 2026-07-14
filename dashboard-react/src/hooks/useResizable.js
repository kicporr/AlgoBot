import { useRef, useEffect, useCallback } from 'react';

export function useResizable(initialSize, minSize, maxSize, axis = 'vertical') {
  const sizeRef = useRef(initialSize);
  const dragging = useRef(false);
  const startPos = useRef(0);
  const startSize = useRef(0);
  const handleRef = useRef(null);

  const getSize = useCallback(() => sizeRef.current, []);

  useEffect(() => {
    const handle = handleRef.current;
    if (!handle) return;

    const onDown = (e) => {
      e.preventDefault();
      dragging.current = true;
      startPos.current = axis === 'vertical' ? e.clientY : e.clientX;
      startSize.current = sizeRef.current;
      handle.style.cursor = axis === 'vertical' ? 'row-resize' : 'col-resize';
      document.body.style.userSelect = 'none';
      document.body.style.cursor = axis === 'vertical' ? 'row-resize' : 'col-resize';
    };

    const onMove = (e) => {
      if (!dragging.current) return;
      const pos = axis === 'vertical' ? e.clientY : e.clientX;
      const delta = axis === 'vertical'
        ? startPos.current - pos  // drag up = larger
        : startPos.current - pos; // drag left = larger
      sizeRef.current = Math.min(maxSize, Math.max(minSize, startSize.current - delta));
      handle.dispatchEvent(new CustomEvent('resize', { detail: sizeRef.current }));
    };

    const onUp = () => {
      if (!dragging.current) return;
      dragging.current = false;
      handle.style.cursor = axis === 'vertical' ? 'ns-resize' : 'ew-resize';
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
    };

    handle.style.cursor = axis === 'vertical' ? 'ns-resize' : 'ew-resize';
    handle.addEventListener('mousedown', onDown);
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      handle.removeEventListener('mousedown', onDown);
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [axis, minSize, maxSize]);

  return { handleRef, getSize };
}
